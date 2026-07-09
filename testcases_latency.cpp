/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstring>

#include "testcase.h"
#include "memcpy.h"
#include "kernels.cuh"
#include "output.h"

// Message-size latency sweep implementation.
//
// Sizes are swept in powers of two over [minMsgSize, maxMsgSize] (CLI
// --minMsgSize/--maxMsgSize, default 1KiB..2MiB). One latency matrix (us per
// message) is emitted per size. Loop/iteration counts are tiered by size so
// that each timed window stays in the millisecond range: CUDA event
// resolution (~0.5us) then contributes <0.1% error, while the number of
// commands pre-enqueued behind the stream blocker stays bounded.

static std::vector<unsigned long long> messageSizeSweep() {
    std::vector<unsigned long long> sizes;
    unsigned long long s = minMsgSize;
    while (s <= maxMsgSize) {
        sizes.push_back(s);
        if (s > maxMsgSize / 2) {
            break;  // s <<= 1 would overflow or exceed the bound
        }
        s <<= 1;
    }
    return sizes;
}

static std::string msgSizeLabel(unsigned long long bytes) {
    std::stringstream ss;
    if (bytes >= _MiB && bytes % _MiB == 0) {
        ss << (bytes / _MiB) << "MiB";
    } else if (bytes >= 1024 && bytes % 1024 == 0) {
        ss << (bytes / 1024) << "KiB";
    } else {
        ss << bytes << "B";
    }
    return ss.str();
}

// Back-to-back cuMemcpyAsync count per timed window
static unsigned long long ceLatencyLoopCount(unsigned long long msgSize) {
    if (msgSize <= 64 * 1024ULL) return 512;
    if (msgSize <= 512 * 1024ULL) return 256;
    return 128;
}

// Timed ping-pong rounds
static unsigned int pingPongIters(unsigned long long msgSize) {
    if (msgSize <= 64 * 1024ULL) return 2000;
    if (msgSize <= 512 * 1024ULL) return 800;
    return 200;
}

// Copy-kernel block-count candidates per side. A single block avoids grid
// syncs on the critical path but is single-SM store-throughput-bound; more
// blocks speed up the copy phase but add two grid syncs per one-way leg.
// The crossover depends on the interconnect, so mid-range sizes measure all
// candidates and the fastest (minimum) latency is reported.
static std::vector<unsigned int> pingPongBlockCandidates(unsigned long long msgSize) {
    if (msgSize <= 32 * 1024ULL) return {1};
    if (msgSize <= 512 * 1024ULL) return {1, 4, 8};
    return {8, 16};
}

static const unsigned int pingPongWarmupRounds = 32;

static void ceMessageLatencySweep(const std::string &key, bool isRead) {
    for (unsigned long long msgSize : messageSizeSweep()) {
        const std::string label = msgSizeLabel(msgSize);
        PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key + "_" + label, perfFormatter, LATENCY_US);
        MemcpyOperation memcpyInstance(ceLatencyLoopCount(msgSize), new MemcpyInitiatorCE(),
                                       isRead ? PREFER_DST_CONTEXT : PREFER_SRC_CONTEXT);

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
                if (peerDeviceId == srcDeviceId) {
                    continue;
                }

                DeviceBuffer srcBuffer(msgSize, srcDeviceId);
                DeviceBuffer peerBuffer(msgSize, peerDeviceId);

                if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                    continue;
                }

                // doMemcpy returns GB/s over the message size; invert to us per message
                double bandwidth = isRead ? memcpyInstance.doMemcpy(peerBuffer, srcBuffer)
                                          : memcpyInstance.doMemcpy(srcBuffer, peerBuffer);
                if (bandwidth > 0.0) {
                    latencyValues.value(srcDeviceId, peerDeviceId) = (double) msgSize / (bandwidth * 1000.0);
                }
            }
        }

        output->addTestcase(key + "_" + label, NVB_RUNNING);
        output->addTestcaseResults(latencyValues,
            std::string("memcpy CE GPU(row) ") + (isRead ? "reads from" : "writes to") +
            " GPU(column) per-message latency (us), message size " + label);
    }

    if (jsonOutput) {
        // Finalize the parent sweep entry (per-size sub-entries got their own status)
        output->setTestcaseStatusAndAddIfNeeded(key, NVB_PASSED);
    }
}

void DeviceToDeviceMessageLatencyWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    ceMessageLatencySweep(key, false);
}

void DeviceToDeviceMessageLatencyReadCE::run(unsigned long long size, unsigned long long loopCount) {
    ceMessageLatencySweep(key, true);
}

// Fill a device buffer with the deterministic xorshift pattern (tiled in 2MiB
// chunks), used to verify the ping-pong echo end-to-end on the host
static void fillDevicePattern(const DeviceBuffer &buf, const std::vector<unsigned int> &pattern) {
    CU_ASSERT(cuCtxSetCurrent(buf.getPrimaryCtx()));
    size_t remaining = buf.getBufferSize();
    size_t offset = 0;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, (size_t) _2MiB);
        CU_ASSERT(cuMemcpyHtoD(buf.getBuffer() + offset, pattern.data(), chunk));
        offset += chunk;
        remaining -= chunk;
    }
}

void DeviceToDeviceMessageLatencyPingPongSM::run(unsigned long long size, unsigned long long loopCount) {
    std::vector<unsigned int> pattern(_2MiB / sizeof(unsigned int));
    xorshift2MBPattern(pattern.data(), 0xBAADF00D);

    for (unsigned long long msgSize : messageSizeSweep()) {
        const std::string label = msgSizeLabel(msgSize);
        PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key + "_" + label, perfFormatter, LATENCY_US);
        const unsigned int iters = pingPongIters(msgSize);
        const std::vector<unsigned int> blockCandidates = pingPongBlockCandidates(msgSize);

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
                if (peerDeviceId == srcDeviceId) {
                    continue;
                }

                DeviceBuffer srcBuffer(msgSize, srcDeviceId);
                DeviceBuffer echoBuffer(msgSize, srcDeviceId);
                DeviceBuffer recvBuffer(msgSize, peerDeviceId);

                if (!srcBuffer.enablePeerAcess(recvBuffer)) {
                    continue;
                }

                fillDevicePattern(srcBuffer, pattern);
                CU_ASSERT(cuCtxSetCurrent(echoBuffer.getPrimaryCtx()));
                CU_ASSERT(cuMemsetD8(echoBuffer.getBuffer(), 0, msgSize));
                CU_ASSERT(cuCtxSetCurrent(recvBuffer.getPrimaryCtx()));
                CU_ASSERT(cuMemsetD8(recvBuffer.getBuffer(), 0, msgSize));
                CU_ASSERT(cuCtxSynchronize());

                // Auto-tune the copy grid: report the fastest block-count candidate
                double bestLatencyUs = 0.0;
                for (unsigned int numBlocks : blockCandidates) {
                    PerformanceStatistic latencyStat;
                    for (unsigned int n = 0; n < averageLoopCount; n++) {
                        double latencyUs = pingPongOneWayLatencyUs(srcDeviceId, peerDeviceId,
                            srcBuffer.getBuffer(), echoBuffer.getBuffer(), recvBuffer.getBuffer(),
                            msgSize, iters, pingPongWarmupRounds, numBlocks);
                        latencyStat(latencyUs);
                        VERBOSE << "\tSample " << n << " (" << numBlocks << " blocks): "
                                << srcBuffer.getBufferString() << " <-> "
                                << recvBuffer.getBufferString() << " (" << label << "): "
                                << std::fixed << std::setprecision(3) << latencyUs << " us\n";
                    }
                    double candidateUs = latencyStat.returnAppropriateMetric();
                    if (bestLatencyUs == 0.0 || candidateUs < bestLatencyUs) {
                        bestLatencyUs = candidateUs;
                    }
                }

                if (!skipVerification) {
                    // The responder echoed the received data back: the echo buffer
                    // must match the original pattern after the last round
                    std::vector<unsigned char> echoHost(msgSize);
                    CU_ASSERT(cuCtxSetCurrent(echoBuffer.getPrimaryCtx()));
                    CU_ASSERT(cuMemcpyDtoH(echoHost.data(), echoBuffer.getBuffer(), msgSize));
                    size_t offset = 0;
                    while (offset < msgSize) {
                        size_t chunk = std::min((size_t) msgSize - offset, (size_t) _2MiB);
                        ASSERT(std::memcmp(echoHost.data() + offset, pattern.data(), chunk) == 0);
                        offset += chunk;
                    }
                }

                latencyValues.value(srcDeviceId, peerDeviceId) = bestLatencyUs;
            }
        }

        output->addTestcase(key + "_" + label, NVB_RUNNING);
        output->addTestcaseResults(latencyValues,
            "SM ping-pong one-way latency GPU(row) -> GPU(column) (us), message size " + label);
    }

    if (jsonOutput) {
        // Finalize the parent sweep entry (per-size sub-entries got their own status)
        output->setTestcaseStatusAndAddIfNeeded(key, NVB_PASSED);
    }
}
