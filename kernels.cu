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

#include "kernels.cuh"

#include <cooperative_groups.h>

__global__ void simpleCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t offset = idx * sizeof(uint4);
        uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)dst + offset);
        uint4* src_uint4 = reinterpret_cast<uint4*>((char*)src + offset);
        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}

__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* dst, uint4* src, size_t chunkSizeInElement) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;

        while (cdst < dstBigEnd) {
            uint4 pipe_0 = *csrc; csrc += totalThreadCount;
            uint4 pipe_1 = *csrc; csrc += totalThreadCount;
            uint4 pipe_2 = *csrc; csrc += totalThreadCount;
            uint4 pipe_3 = *csrc; csrc += totalThreadCount;
            uint4 pipe_4 = *csrc; csrc += totalThreadCount;
            uint4 pipe_5 = *csrc; csrc += totalThreadCount;
            uint4 pipe_6 = *csrc; csrc += totalThreadCount;
            uint4 pipe_7 = *csrc; csrc += totalThreadCount;
            uint4 pipe_8 = *csrc; csrc += totalThreadCount;
            uint4 pipe_9 = *csrc; csrc += totalThreadCount;
            uint4 pipe_10 = *csrc; csrc += totalThreadCount;
            uint4 pipe_11 = *csrc; csrc += totalThreadCount;

            *cdst = pipe_0; cdst += totalThreadCount;
            *cdst = pipe_1; cdst += totalThreadCount;
            *cdst = pipe_2; cdst += totalThreadCount;
            *cdst = pipe_3; cdst += totalThreadCount;
            *cdst = pipe_4; cdst += totalThreadCount;
            *cdst = pipe_5; cdst += totalThreadCount;
            *cdst = pipe_6; cdst += totalThreadCount;
            *cdst = pipe_7; cdst += totalThreadCount;
            *cdst = pipe_8; cdst += totalThreadCount;
            *cdst = pipe_9; cdst += totalThreadCount;
            *cdst = pipe_10; cdst += totalThreadCount;
            *cdst = pipe_11; cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

// This kernel performs a split warp copy, alternating copy directions across warps.
__global__ void splitWarpCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int globalWarpId = idx / warpSize;
        unsigned int warpLaneId = idx % warpSize;
        uint4* dst_uint4;
        uint4* src_uint4;

        // alternate copy directions across warps
        if (globalWarpId & 0x1) {
            // odd warp
            dst_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = src + (globalWarpId * warpSize + warpLaneId);
        } else {
            // even warp
            dst_uint4 = src + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
        }

        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}

__global__ void ptrChasingKernel(struct LatencyNode *data, size_t size, unsigned int accesses, unsigned int targetBlock) {
    struct LatencyNode *p = data;
    if (blockIdx.x != targetBlock) return;
    for (auto i = 0; i < accesses; ++i) {
        p = p->next;
    }

    // avoid compiler optimization
    if (p == nullptr) {
        __trap();
    }
}

static __device__ __noinline__
void mc_st_u32(unsigned int *dst, unsigned int v) {
#if __CUDA_ARCH__ >= 900
    asm volatile ("multimem.st.u32 [%0], %1;" :: "l"(dst), "r" (v));
#endif
}

static __device__ __noinline__
void mc_ld_u32(unsigned int *dst, const unsigned int *src) {
#if __CUDA_ARCH__ >= 900
     asm volatile ("multimem.ld_reduce.and.b32 %0, [%1];" : "=r"((*dst)) : "l" (src));
#endif
}

// Writes from regular memory to multicast memory
__global__ void multicastCopyKernel(unsigned long long loopCount, unsigned int* __restrict__ dst, unsigned int* __restrict__ src, size_t nElems) {
    const size_t totalThreadCount = blockDim.x * gridDim.x;
    const size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int* const enddst = dst + nElems;
    dst += offset;
    src += offset;

    for (unsigned int i = 0; i < loopCount; i++) {
        // Reset pointers to src and dst chunks.
        unsigned int* cur_src_ptr = src;
        unsigned int* cur_dst_ptr = dst;
        #pragma unroll(12)
        while (cur_dst_ptr < enddst) {
            mc_st_u32(cur_dst_ptr, *cur_src_ptr);
            cur_dst_ptr += totalThreadCount;
            cur_src_ptr += totalThreadCount;
        }
    }
}

double latencyPtrChaseKernel(const int srcId, void* data, size_t size, unsigned long long latencyMemAccessCnt, unsigned smCount) {
    CUstream stream;
    int device, clock_rate_khz;
    double latencySum = 0.0f, finalLatencyPerAccessNs = 0.0;
    CUcontext srcCtx;
    cudaEvent_t start, end;
    float latencyMs = 0;

    CUDA_ASSERT(cudaEventCreate(&start));
    CUDA_ASSERT(cudaEventCreate(&end));

    CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, srcId));
    CU_ASSERT(cuCtxSetCurrent(srcCtx));

    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CU_ASSERT(cuCtxGetDevice(&device));
    CU_ASSERT(cuDeviceGetAttribute(&clock_rate_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    for (int targetBlock = 0; targetBlock < smCount; ++targetBlock) {
        CUDA_ASSERT(cudaEventRecord(start, stream));
        ptrChasingKernel <<< smCount, 1, 0, stream>>> ((struct LatencyNode*) data, size, latencyMemAccessCnt / smCount, targetBlock);
        CUDA_ASSERT(cudaEventRecord(end, stream));
        CUDA_ASSERT(cudaGetLastError());
        CU_ASSERT(cuStreamSynchronize(stream));
        cudaEventElapsedTime(&latencyMs, start, end);
        latencySum += (latencyMs / 1000);
    }
    finalLatencyPerAccessNs = (latencySum * 1.0E9) / (latencyMemAccessCnt);

    CUDA_ASSERT(cudaEventDestroy(start));
    CUDA_ASSERT(cudaEventDestroy(end));

    return finalLatencyPerAccessNs;
}

size_t copyKernel(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;

    // If the user provided buffer size is samller than default buffer size,
    // we use the simple copy kernel for our bandwidth test.
    // This is done so that no trucation of the buffer size occurs.
    // Please note that to achieve peak bandwidth, it is suggested to use the
    // default buffer size, which in turn triggers the use of the optimized
    // kernel.
    if (desc.copySize < (smallBufferThreshold * _MiB)) {
        // copy size is rounded down to 16 bytes
        unsigned int numUint4 = desc.copySize / sizeof(uint4);
        // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
        dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));
        dim3 grid(numUint4/block.x);
        simpleCopyKernel <<<grid, block, 0 , desc.stream>>> (desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src);
        return numUint4 * sizeof(uint4);
    }

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = desc.copySize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    stridingMemcpyKernel<<<gridDim, blockDim, 0, desc.stream>>> (totalThreadCount, desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src, chunkSizeInElement);

    return sizeInElement * sizeof(uint4);
}

size_t copyKernelSplitWarp(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    // copy size is rounded down to 16 bytes
    unsigned int numUint4 = desc.copySize / sizeof(uint4);

    // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
    dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));
    dim3 grid(numUint4/block.x);
    splitWarpCopyKernel <<<grid, block, 0 , desc.stream>>> (desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src);
    return numUint4 * sizeof(uint4);
}

size_t multicastCopy(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(unsigned);
    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    multicastCopyKernel<<<gridDim, blockDim, 0, stream>>> (loopCount, (unsigned *)dstBuffer, (unsigned *)srcBuffer, sizeInElement);
    return sizeInElement * sizeof(unsigned);
}

__global__ void spinKernelDevice(volatile int *latch, const unsigned long long timeoutClocks) {
    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutMs) {
    int clocksPerMs = 0;
    CUcontext ctx;
    CUdevice dev;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    CU_ASSERT(cuDeviceGetAttribute(&clocksPerMs, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    unsigned long long timeoutClocks = clocksPerMs * timeoutMs;

    spinKernelDevice<<<1, 1, 0, stream>>>(latch, timeoutClocks);

    return CUDA_SUCCESS;
}

__global__ void spinKernelDeviceMultistage(volatile int *latch1, volatile int *latch2, const unsigned long long timeoutClocks) {
    if (latch1) {
        register unsigned long long endTime = clock64() + timeoutClocks;
        while (!*latch1) {
            if (timeoutClocks != ~0ULL && clock64() > endTime) {
                return;
            }
        }

        *latch2 = 1;
    }

    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch2) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

// Implement a 2-stage spin kernel for multi-node synchronization.
// One of the host nodes releases the first latch. Subsequently,
// the second latch is released, that is polled by all other devices
// latch1 argument is optional. If defined, kernel will spin on it until released, and then will release latch2.
// latch2 argument is mandatory. Kernel will spin on it until released.
// timeoutMs argument applies to each stage separately.
// However, since each kernel will spin on only one stage, total runtime is still limited by timeoutMs
CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs) {
    int clocksPerMs = 0;
    CUcontext ctx;
    CUdevice dev;

    ASSERT(latch2 != nullptr);

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    CU_ASSERT(cuDeviceGetAttribute(&clocksPerMs, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    unsigned long long timeoutClocks = clocksPerMs * timeoutMs;

    spinKernelDeviceMultistage<<<1, 1, 0, stream>>>(latch1, latch2, timeoutClocks);

    return CUDA_SUCCESS;
}

__global__ void memsetKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        buf[idx] = pat[idx % num_pattern_elements];
    }
}

// This kernel clears memory locations in the buffer based on warp parity.
// If clearOddWarpIndexed is true, it clears buffer locations indexed by odd warps.
// Otherwise, it clears buffer locations indexed by even warps.
__global__ void memclearKernelByWarpParityDevice(CUdeviceptr buffer, bool clearOddWarpIndexed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint4* buf = reinterpret_cast<uint4*>(buffer);
    unsigned int globalWarpId = idx / warpSize;
    unsigned int thread_idx_in_warp = idx % warpSize;

    if (clearOddWarpIndexed) {
        // clear memory locations in buffer indexed by odd warps
        if (globalWarpId & 0x1) {
            buf[globalWarpId * warpSize + thread_idx_in_warp] = make_uint4(0x0, 0x0, 0x0, 0x0);
        }
    } else {
        // clear memory locations in buffer indexed by even warps
        if (!(globalWarpId & 0x1)) {
            buf[globalWarpId * warpSize + thread_idx_in_warp] = make_uint4(0x0, 0x0, 0x0, 0x0);
        }
    }
}

__global__ void memcmpKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        if (buf[idx] != pat[idx % num_pattern_elements]) {
            if (atomicCAS((int*)errorFlag, 0, 1) == 0) {
                // have the first thread that detects a mismatch print the error message
                printf(" Invalid value when checking the pattern at %p\n", (void*)((char*)buffer));
                printf(" Current offset : %lu \n", idx);
                return;
            }
        }
    }
}

__global__ void multicastMemcmpKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        unsigned buf_val;
        mc_ld_u32(&buf_val, &buf[idx]);
        if (buf_val != pat[idx % num_pattern_elements]) {
            if (atomicCAS((int*)errorFlag, 0, 1) == 0) {
                // have the first thread that detects a mismatch print the error message
                printf(" Invalid value when checking the pattern at %p\n", (void*)((char*)buffer));
                printf(" Current offset : %lu \n", idx);
                return;
            }
        }
    }
}

CUresult memsetKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    memsetKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult memclearKernelByWarpParity(CUstream stream, CUdeviceptr buffer, size_t size, bool clearOddWarpIndexed) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    // copy size is rounded down to 16 bytes
    unsigned int numUint4 = size / sizeof(uint4);

    // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
    dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));

    dim3 grid(numUint4/block.x);
    memclearKernelByWarpParityDevice <<<grid, block, 0 , stream>>> (buffer, clearOddWarpIndexed);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult memcmpKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    memcmpKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements, errorFlag);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult multicastMemcmpKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    multicastMemcmpKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements, errorFlag);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Ping-pong message latency kernel
//
// Two instances of this kernel run persistently, one per GPU of a pair.
// The initiator copies the message into the responder's receive buffer via
// P2P stores, fences system-wide, then sets a flag in the responder's memory.
// The responder polls its local flag (remote writes, local reads only),
// echoes the received data back into the initiator's echo buffer and sets the
// initiator's flag. One-way latency = measured round-trip time / 2.
// Timing uses %globaltimer (ns) on the initiator only, so no cross-GPU clock
// synchronization is required; warmup rounds are excluded.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void pingPongSync(cooperative_groups::grid_group &grid, bool multiBlock) {
    if (multiBlock) {
        grid.sync();
    } else {
        __syncthreads();
    }
}

__device__ __forceinline__ void pingPongGridCopy(uint4 *dst, const uint4 *src, size_t numElems) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < numElems; i += stride) {
        __stcg(dst + i, __ldcg(src + i));
    }
}

// Waits until *flag >= r; the timeout budget applies to this single wait.
// Returns false (and sets the error flag) on timeout.
__device__ __forceinline__ bool pingPongWaitFlag(volatile unsigned int *flag, unsigned int r,
                                                 unsigned long long timeoutNs, volatile int *errorFlagOut) {
    const unsigned long long waitStart = globalTimerNs();
    while (*flag < r) {
        if (globalTimerNs() - waitStart > timeoutNs) {
            *errorFlagOut = 1;
            return false;
        }
    }
    return true;
}

__global__ void pingPongKernel(uint4 *writeDst, const uint4 *readSrc, size_t numElems,
                               volatile unsigned int *localFlag, volatile unsigned int *remoteFlag,
                               unsigned int totalRounds, unsigned int warmupRounds, int isInitiator,
                               unsigned long long timeoutNs,
                               unsigned long long *elapsedNsOut, volatile int *errorFlagOut) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const bool multiBlock = gridDim.x > 1;
    const bool leader = (blockIdx.x == 0 && threadIdx.x == 0);
    unsigned long long t0 = 0;

    for (unsigned int r = 1; r <= totalRounds; r++) {
        if (isInitiator) {
            if (leader && r == warmupRounds + 1) {
                t0 = globalTimerNs();
            }
            pingPongGridCopy(writeDst, readSrc, numElems);
            // Make the payload globally visible before the flag store below;
            // grid-wide barrier orders all threads' fenced stores before it
            __threadfence_system();
            pingPongSync(grid, multiBlock);
            if (leader) {
                *remoteFlag = r;
                pingPongWaitFlag(localFlag, r, timeoutNs, errorFlagOut);
                __threadfence_system();
            }
            pingPongSync(grid, multiBlock);
            // Abort promptly after a timeout instead of burning the remaining rounds
            if (*errorFlagOut) {
                break;
            }
        } else {
            if (leader) {
                pingPongWaitFlag(localFlag, r, timeoutNs, errorFlagOut);
                __threadfence_system();
            }
            pingPongSync(grid, multiBlock);
            if (*errorFlagOut) {
                break;
            }
            pingPongGridCopy(writeDst, readSrc, numElems);
            __threadfence_system();
            pingPongSync(grid, multiBlock);
            if (leader) {
                *remoteFlag = r;
            }
        }
    }

    if (isInitiator && leader && elapsedNsOut != nullptr) {
        *elapsedNsOut = globalTimerNs() - t0;
    }
}

double pingPongOneWayLatencyUs(int initiatorDev, int responderDev,
                               CUdeviceptr initiatorSrc, CUdeviceptr initiatorEcho, CUdeviceptr responderRecv,
                               size_t msgSize, unsigned int iters, unsigned int warmupRounds, unsigned int numBlocks,
                               unsigned long long timeoutNs) {
    size_t numElems = msgSize / sizeof(uint4);
    ASSERT(numElems > 0);
    ASSERT(iters > 0);
    unsigned int totalRounds = iters + warmupRounds;

    int coopSupported = 0;
    CUDA_ASSERT(cudaDeviceGetAttribute(&coopSupported, cudaDevAttrCooperativeLaunch, initiatorDev));
    ASSERT(coopSupported);
    CUDA_ASSERT(cudaDeviceGetAttribute(&coopSupported, cudaDevAttrCooperativeLaunch, responderDev));
    ASSERT(coopSupported);

    cudaStream_t streamInit, streamResp;
    unsigned int *flagInit, *flagResp;
    unsigned long long *elapsedOut;
    int *errInit, *errResp;

    CUDA_ASSERT(cudaSetDevice(responderDev));
    CUDA_ASSERT(cudaStreamCreateWithFlags(&streamResp, cudaStreamNonBlocking));
    CUDA_ASSERT(cudaMalloc(&flagResp, sizeof(unsigned int)));
    CUDA_ASSERT(cudaMalloc(&errResp, sizeof(int)));
    CUDA_ASSERT(cudaMemset(flagResp, 0, sizeof(unsigned int)));
    CUDA_ASSERT(cudaMemset(errResp, 0, sizeof(int)));
    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaSetDevice(initiatorDev));
    CUDA_ASSERT(cudaStreamCreateWithFlags(&streamInit, cudaStreamNonBlocking));
    CUDA_ASSERT(cudaMalloc(&flagInit, sizeof(unsigned int)));
    CUDA_ASSERT(cudaMalloc(&errInit, sizeof(int)));
    CUDA_ASSERT(cudaMalloc(&elapsedOut, sizeof(unsigned long long)));
    CUDA_ASSERT(cudaMemset(flagInit, 0, sizeof(unsigned int)));
    CUDA_ASSERT(cudaMemset(errInit, 0, sizeof(int)));
    CUDA_ASSERT(cudaMemset(elapsedOut, 0, sizeof(unsigned long long)));
    CUDA_ASSERT(cudaDeviceSynchronize());

    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);

    // Responder arguments: echo local recvBuf into the initiator's echo buffer
    uint4 *rWriteDst = (uint4 *)initiatorEcho;
    const uint4 *rReadSrc = (const uint4 *)responderRecv;
    volatile unsigned int *rLocalFlag = flagResp;
    volatile unsigned int *rRemoteFlag = flagInit;
    int rIsInitiator = 0;
    unsigned long long *rElapsedOut = nullptr;
    void *argsResp[] = {&rWriteDst, &rReadSrc, &numElems, &rLocalFlag, &rRemoteFlag,
                        &totalRounds, &warmupRounds, &rIsInitiator, &timeoutNs, &rElapsedOut, &errResp};

    // Initiator arguments: push local srcBuf into the responder's receive buffer
    uint4 *iWriteDst = (uint4 *)responderRecv;
    const uint4 *iReadSrc = (const uint4 *)initiatorSrc;
    volatile unsigned int *iLocalFlag = flagInit;
    volatile unsigned int *iRemoteFlag = flagResp;
    int iIsInitiator = 1;
    void *argsInit[] = {&iWriteDst, &iReadSrc, &numElems, &iLocalFlag, &iRemoteFlag,
                        &totalRounds, &warmupRounds, &iIsInitiator, &timeoutNs, &elapsedOut, &errInit};

    CUDA_ASSERT(cudaSetDevice(responderDev));
    CUDA_ASSERT(cudaLaunchCooperativeKernel((void *)pingPongKernel, gridDim, blockDim, argsResp, 0, streamResp));
    CUDA_ASSERT(cudaSetDevice(initiatorDev));
    CUDA_ASSERT(cudaLaunchCooperativeKernel((void *)pingPongKernel, gridDim, blockDim, argsInit, 0, streamInit));

    CUDA_ASSERT(cudaStreamSynchronize(streamInit));
    CUDA_ASSERT(cudaStreamSynchronize(streamResp));

    unsigned long long elapsedNs = 0;
    int initTimedOut = 0, respTimedOut = 0;
    CUDA_ASSERT(cudaMemcpy(&elapsedNs, elapsedOut, sizeof(elapsedNs), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&initTimedOut, errInit, sizeof(initTimedOut), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaSetDevice(responderDev));
    CUDA_ASSERT(cudaMemcpy(&respTimedOut, errResp, sizeof(respTimedOut), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(flagResp));
    CUDA_ASSERT(cudaFree(errResp));
    CUDA_ASSERT(cudaStreamDestroy(streamResp));
    CUDA_ASSERT(cudaSetDevice(initiatorDev));
    CUDA_ASSERT(cudaFree(flagInit));
    CUDA_ASSERT(cudaFree(errInit));
    CUDA_ASSERT(cudaFree(elapsedOut));
    CUDA_ASSERT(cudaStreamDestroy(streamInit));

    if (initTimedOut || respTimedOut) {
        throw std::string("Ping-pong latency kernel timed out waiting for the peer GPU (devices ") +
            std::to_string(initiatorDev) + " <-> " + std::to_string(responderDev) + ")";
    }

    return (double)elapsedNs / (2.0 * iters) / 1000.0;
}

// ---------------------------------------------------------------------------
// SM copy message latency kernel (NCCL-style load/store data path)
//
// Runs on the initiating GPU only. Each iteration copies the whole message
// with grid-strided SM loads/stores and ends with __threadfence_system() plus
// a grid-wide barrier, mirroring NCCL's per-chunk "copy + fence (+ flag)"
// pattern. With a remote destination (write/push) this measures the sender's
// per-message issue+drain cost; with a remote source (read/pull) it measures
// the load-round-trip-bound pull cost. Timed on-device with %globaltimer.
// ---------------------------------------------------------------------------

__global__ void smMsgLatencyKernel(uint4 *dst, const uint4 *src, size_t numElems,
                                   unsigned int totalIters, unsigned int warmupIters,
                                   unsigned long long *elapsedNsOut) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const bool multiBlock = gridDim.x > 1;
    const bool leader = (blockIdx.x == 0 && threadIdx.x == 0);
    unsigned long long t0 = 0;

    for (unsigned int i = 1; i <= totalIters; i++) {
        if (leader && i == warmupIters + 1) {
            t0 = globalTimerNs();
        }
        pingPongGridCopy(dst, src, numElems);
        __threadfence_system();
        pingPongSync(grid, multiBlock);
    }

    if (leader && elapsedNsOut != nullptr) {
        *elapsedNsOut = globalTimerNs() - t0;
    }
}

double smMessageLatencyUs(int execDev, CUdeviceptr dst, CUdeviceptr src,
                          size_t msgSize, unsigned int iters, unsigned int warmupIters,
                          unsigned int numBlocks) {
    size_t numElems = msgSize / sizeof(uint4);
    ASSERT(numElems > 0);
    ASSERT(iters > 0);
    unsigned int totalIters = iters + warmupIters;

    int coopSupported = 0;
    CUDA_ASSERT(cudaDeviceGetAttribute(&coopSupported, cudaDevAttrCooperativeLaunch, execDev));
    ASSERT(coopSupported);

    cudaStream_t stream;
    unsigned long long *elapsedOut;

    CUDA_ASSERT(cudaSetDevice(execDev));
    CUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_ASSERT(cudaMalloc(&elapsedOut, sizeof(unsigned long long)));
    CUDA_ASSERT(cudaMemset(elapsedOut, 0, sizeof(unsigned long long)));

    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);

    uint4 *kDst = (uint4 *)dst;
    const uint4 *kSrc = (const uint4 *)src;
    void *args[] = {&kDst, &kSrc, &numElems, &totalIters, &warmupIters, &elapsedOut};

    CUDA_ASSERT(cudaLaunchCooperativeKernel((void *)smMsgLatencyKernel, gridDim, blockDim, args, 0, stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));

    unsigned long long elapsedNs = 0;
    CUDA_ASSERT(cudaMemcpy(&elapsedNs, elapsedOut, sizeof(elapsedNs), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(elapsedOut));
    CUDA_ASSERT(cudaStreamDestroy(stream));

    return (double)elapsedNs / iters / 1000.0;
}

void preloadKernels(int deviceCount) {
    cudaFuncAttributes unused;
#ifdef MULTINODE
    // In multinode mode, only test the local GPU assigned to the process.
    const int startDevice = localDevice;
    const int endDevice = localDevice + 1;
#else
    // In single-node mode, preload kernels on all GPUs
    const int startDevice = 0;
    const int endDevice = deviceCount;
#endif
    for (int iDev = startDevice; iDev < endDevice; iDev++) {
        cudaSetDevice(iDev);
        cudaFuncGetAttributes(&unused, &stridingMemcpyKernel);
        cudaFuncGetAttributes(&unused, &spinKernelDevice);
        cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
        cudaFuncGetAttributes(&unused, &simpleCopyKernel);
        cudaFuncGetAttributes(&unused, &splitWarpCopyKernel);
        cudaFuncGetAttributes(&unused, &ptrChasingKernel);
        cudaFuncGetAttributes(&unused, &multicastCopyKernel);
        cudaFuncGetAttributes(&unused, &memsetKernelDevice);
        cudaFuncGetAttributes(&unused, &memcmpKernelDevice);
        cudaFuncGetAttributes(&unused, &multicastMemcmpKernelDevice);
        cudaFuncGetAttributes(&unused, &pingPongKernel);
        cudaFuncGetAttributes(&unused, &smMsgLatencyKernel);
    }
}



