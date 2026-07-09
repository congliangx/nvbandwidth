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

#include <boost/program_options.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <iostream>

#include "environment.h"
#include "json_output.h"
#include "kernels.cuh"
#include "output.h"
#include "testcase.h"
#include "version.h"
#include "inline_common.h"

namespace opt = boost::program_options;
int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool verbose;
bool shouldOutput = true;
bool disableAffinity;
bool skipVerification;
bool useMean;
bool perfFormatter;
bool useHugePages;
long long targetNumPairs;
unsigned long long minMsgSize;
unsigned long long maxMsgSize;

Verbosity VERBOSE(verbose);
Verbosity OUTPUT(shouldOutput);

// Device ordinal of the GPU owned by the process
int localDevice = 0;
int localRank = 0;
// Process rank within one OS
int worldRank = 0;
int worldSize = 0;
bool jsonOutput;
Output *output;

std::unique_ptr<Environment> env;

// Define testcases here
std::vector<Testcase*> createNodeTestcases() {
    std::vector<Testcase*> tests;
    if (worldRank == 0 && localRank == 0) {
         tests.insert(tests.end(), {
        // Base tests that always run, by a single node/process only.
        new HostToDeviceCE(),
        new DeviceToHostCE(),
        new HostToDeviceBidirCE(),
        new DeviceToHostBidirCE(),
        new DeviceToDeviceReadCE(),
        new DeviceToDeviceWriteCE(),
        new DeviceToDeviceBidirReadCE(),
        new DeviceToDeviceBidirWriteCE(),
        new AllToHostCE(),
        new AllToHostBidirCE(),
        new HostToAllCE(),
        new HostToAllBidirCE(),
        new AllToOneWriteCE(),
        new AllToOneReadCE(),
        new OneToAllWriteCE(),
        new OneToAllReadCE(),
        new HostToDeviceSM(),
        new DeviceToHostSM(),
        new HostToDeviceBidirSM(),
        new DeviceToHostBidirSM(),
        new DeviceToDeviceReadSM(),
        new DeviceToDeviceWriteSM(),
        new DeviceToDeviceBidirReadSM(),
        new DeviceToDeviceBidirWriteSM(),
        new AllToHostSM(),
        new AllToHostBidirSM(),
        new HostToAllSM(),
        new HostToAllBidirSM(),
        new AllToOneWriteSM(),
        new AllToOneReadSM(),
        new OneToAllWriteSM(),
        new OneToAllReadSM(),
        new HostDeviceLatencySM(),
        new DeviceToDeviceLatencySM(),
        new DeviceLocalCopy(),
        new DeviceToDeviceMessageLatencyWriteCE(),
        new DeviceToDeviceMessageLatencyReadCE(),
        new DeviceToDeviceMessageLatencyPingPongSM()
      });
    }
#ifdef MULTINODE
    // Add multinode tests only if we're running in multinode mode
    if (env->getSize() > 1) {  // More than one process means we're in multinode mode
        tests.insert(tests.end(), {
            new MultinodeDeviceToDeviceReadCE(),
            new MultinodeDeviceToDeviceWriteCE(),
            new MultinodeDeviceToDeviceBidirReadCE(),
            new MultinodeDeviceToDeviceBidirWriteCE(),
            new MultinodeDeviceToDeviceReadSM(),
            new MultinodeDeviceToDeviceWriteSM(),
            new MultinodeDeviceToDeviceBidirReadSM(),
            new MultinodeDeviceToDeviceBidirWriteSM(),
            new MultinodeAllToOneWriteSM(),
            new MultinodeAllFromOneReadSM(),
            new MultinodeBroadcastOneToAllSM(),
            new MultinodeBroadcastAllToAllSM(),
            new MultinodeBisectWriteCE()
        });
    }
#endif
    return tests;
}
Testcase* findTestcase(std::vector<Testcase*> &testcases, std::string id) {
    // Check if testcase ID is index
    char* p;
    long index = strtol(id.c_str(), &p, 10);
    if (*p) {
        // Conversion failed so key is ID
        auto it = find_if(testcases.begin(), testcases.end(), [&id](Testcase* test) {return test->testKey() == id;});
        if (it != testcases.end()) {
            return testcases.at(std::distance(testcases.begin(), it));
        } else {
            throw "Testcase " + id + " not found!";
        }
    } else {
        // ID is index
        if (index < 0 || index >= static_cast<long>(testcases.size())) throw "Testcase index " + id + " out of bound!";
        return testcases.at(index);
    }
}

std::vector<std::string> expandTestcases(std::vector<Testcase*> &testcases, std::vector<std::string> prefixes) {
    std::vector<std::string> testcasesToRun;
    for (auto testcase : testcases) {
         auto it = find_if(prefixes.begin(), prefixes.end(), [&testcase](std::string prefix) {return testcase->testKey().compare(0, prefix.size(), prefix) == 0;});
            if (it != prefixes.end()) {
                testcasesToRun.push_back(testcase->testKey());
            }
    }
    return testcasesToRun;
}

void runTestcase(std::vector<Testcase*> &testcases, const std::string &testcaseID) {
    Testcase* test{nullptr};
    try {
        test = findTestcase(testcases, testcaseID);
    } catch (std::string &s) {
        output->addTestcase(testcaseID, "ERROR", s);
        return;
    }

    try {
        if (!test->filter()) {
            output->addTestcase(test->testKey(), NVB_WAIVED);
            return;
        }

        output->addTestcase(test->testKey(), NVB_RUNNING);

        // Run the testcase
        if (test->testKey() == "host_device_latency_sm" || test->testKey() == "device_to_device_latency_sm") {
            // use fixd-size buffer for latency tests
            test->run(2 * _MiB, loopCount);
        } else {
            test->run(bufferSize * _MiB, loopCount);
        }
    } catch (std::string &s) {
        output->setTestcaseStatusAndAddIfNeeded(test->testKey(), NVB_ERROR_STATUS, s);
    }
}

int main(int argc, char **argv) {
    env = Environment::create(argc, argv);
    env->initialize(argc, argv);

#ifdef MULTINODE
    worldSize = env->getSize();
    worldRank = env->getRank();
    localRank = env->getLocalRank();
    // Avoid excessive output by limit output to rank 0
    shouldOutput = (worldRank == 0);
#else
    worldSize = deviceCount;
#endif

    std::vector<Testcase*> testcases = createNodeTestcases();
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> testcasePrefixes;
    output = new Output();

    // Args parsing
    opt::options_description visible_opts("nvbandwidth CLI");
    visible_opts.add_options()
        ("help,h", "Produce help message")
        ("bufferSize,b", opt::value<unsigned long long int>(&bufferSize)->default_value(defaultBufferSize), "Memcpy buffer size in MiB")
        ("list,l", "List available testcases")
        ("testcase,t", opt::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name or index)")
        ("testcasePrefixes,p", opt::value<std::vector<std::string>>(&testcasePrefixes)->multitoken(), "Testcase(s) to run (by prefix))")
        ("verbose,v", opt::bool_switch(&verbose)->default_value(false), "Verbose output")
        ("skipVerification,s", opt::bool_switch(&skipVerification)->default_value(false), "Skips data verification after copy")
        ("disableAffinity,d", opt::bool_switch(&disableAffinity)->default_value(false), "Disable automatic CPU affinity control")
        ("testSamples,i", opt::value<unsigned int>(&averageLoopCount)->default_value(defaultAverageLoopCount), "Iterations of the benchmark")
        ("targetNumPairs,P",  opt::value<long long>(&targetNumPairs)->default_value(-1), "Target pairs for multinode device-to-device tests.")
        ("useMean,m", opt::bool_switch(&useMean)->default_value(false), "Use mean instead of median for results")
        ("useHugePages,H",  opt::bool_switch(&useHugePages)->default_value(false), "Use huge pages for host allocations.")
        ("minMsgSize", opt::value<unsigned long long int>(&minMsgSize)->default_value(1024), "Minimum message size in bytes for the *_message_latency_* testcases (rounded down to a power of two)")
        ("maxMsgSize", opt::value<unsigned long long int>(&maxMsgSize)->default_value(2 * _MiB), "Maximum message size in bytes for the *_message_latency_* testcases (rounded down to a power of two)")
        ("json,j", opt::bool_switch(&jsonOutput)->default_value(false), "Print output in json format instead of plain text.");

    opt::options_description all_opts("");
    all_opts.add(visible_opts);
    all_opts.add_options()
        ("loopCount", opt::value<unsigned long long int>(&loopCount)->default_value(defaultLoopCount), "Iterations of memcpy to be performed within a test sample")
        ("perfFormatter", opt::bool_switch(&perfFormatter)->default_value(false), "Use perf formatter prefix (&&&& PERF) in output");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, all_opts), vm);
        opt::notify(vm);
    } catch (...) {
        output->addVersionInfo();

        std::stringstream errmsg;
        errmsg << "ERROR: Invalid Arguments " << std::endl;
        for (int i = 0; i < argc; i++) {
            errmsg << argv[i] << " ";
        }
        std::vector<std::string> messageParts;
        std::stringstream buf;
        buf << visible_opts;
        messageParts.emplace_back(errmsg.str());
        messageParts.emplace_back(buf.str());
        output->recordError(messageParts);
        return 1;
    }

    if (jsonOutput) {
        delete output;
        output = new JsonOutput(shouldOutput);
    }

    if (vm.count("help")) {
        OUTPUT << visible_opts << "\n";
        return 0;
    }

    if (vm.count("list")) {
        output->listTestcases(testcases);
        return 0;
    }

    if (testcasePrefixes.size() != 0 && testcasesToRun.size() != 0) {
        output->recordError("You cannot specify both testcase and testcasePrefix options at the same time");
        return 1;
    }

    // Validate targetNumPairs argument
    if (targetNumPairs < -1) {
        std::stringstream errmsg;
        errmsg << "ERROR: Invalid targetNumPairs value: " << targetNumPairs
               << ". Must be -1 (all pairs), 0 (no pairs), or a positive number.";
        output->recordError(errmsg.str());
        return 1;
    }
    // Validate the message-size sweep bounds: at least one uint4, at most
    // 1 GiB, rounded down to powers of two so copy kernels never truncate
    if (minMsgSize < 16 || maxMsgSize < minMsgSize || maxMsgSize > (1ULL << 30)) {
        std::stringstream errmsg;
        errmsg << "ERROR: Invalid message size range [" << minMsgSize << ", " << maxMsgSize
               << "]. minMsgSize must be >= 16 bytes, <= maxMsgSize; maxMsgSize must be <= 1 GiB.";
        output->recordError(errmsg.str());
        return 1;
    }
    auto floorPow2 = [](unsigned long long v) {
        unsigned long long p = 1;
        while (p <= v / 2) p *= 2;
        return p;
    };
    minMsgSize = floorPow2(minMsgSize);
    maxMsgSize = floorPow2(maxMsgSize);

#ifdef MULTINODE
    // In multinode mode, validate against maximum possible pairs
    if (targetNumPairs > 0) {
        long long maxPairs = static_cast<long long>(worldSize) * (worldSize - 1);
        if (targetNumPairs > maxPairs) {
            std::stringstream errmsg;
            errmsg << "ERROR: targetNumPairs (" << targetNumPairs
                   << ") exceeds maximum possible pairs (" << maxPairs
                   << ") for worldSize " << worldSize
                   << ". Use -1 for all pairs or specify a value <= " << maxPairs << ".";
            output->recordError(errmsg.str());
            return 1;
        }
    }
#endif

    CU_ASSERT(cuInit(0));
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    if (bufferSize < defaultBufferSize) {
        output->recordWarning("NOTE: You have chosen a buffer size that is smaller than the default buffer size. It is suggested to use the default buffer size (512MB) to achieve maximal peak bandwidth.");
    }
#ifdef _WIN32
    if (useHugePages) {
        // Disable huge pages on Windows
        output->recordWarning("NOTE: Huge pages are not supported on Windows. The option will be ignored.");
        useHugePages = false;
    }
#else
    if (useHugePages && !hugePagesEnabled()) {
        output->recordWarning("NOTE: Huge pages were requested, but Transparent Huge Pages (THP) are not enabled on this system. The option will be ignored. Enable THP with 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled' if you wish to use this feature.");
    }
#endif

    int cudaVersion;
    CUDA_ASSERT(cudaRuntimeGetVersion(&cudaVersion));
    CU_ASSERT(cuDriverGetVersion(&cudaVersion));

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

    output->addVersionInfo();
    output->addCudaAndDriverInfo(cudaVersion, driverVersion);
    // Print GPU information
    output->recordDevices(worldSize);

    // Early CUDA runtime sanity check - test if we can create contexts and allocate memory
    // This catches driver/runtime issues before they manifest as confusing errors later
#ifdef MULTINODE
    // In multinode mode, only test the local GPU assigned to the process.
    const int testDeviceStart = localDevice;
    const int testDeviceEnd = localDevice + 1;
#else
    // In single-node mode, test all GPUs
    const int testDeviceStart = 0;
    const int testDeviceEnd = deviceCount;
#endif
    for (int i = testDeviceStart; i < testDeviceEnd; i++) {
        cudaError_t err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            output->recordError("CUDA runtime sanity check failed: cudaSetDevice(" + std::to_string(i) + ") returned " + std::string(cudaGetErrorString(err)));
            return 1;
        }

        // Test basic memory allocation to verify that context creation works
        void* testPtr = nullptr;
        err = cudaMalloc(&testPtr, 1024);
        if (err != cudaSuccess) {
            output->recordError("CUDA runtime sanity check failed: cudaMalloc on device " + std::to_string(i) + " returned " + std::string(cudaGetErrorString(err)));
            return 1;
        }
        cudaFree(testPtr);
    }

    if (testcasePrefixes.size() > 0) {
        testcasesToRun = expandTestcases(testcases, testcasePrefixes);
        if (testcasesToRun.size() == 0) {
            output->recordError("Specified list of testcase prefixes did not match any testcases");
            return 1;
        }
    }

    // This triggers the loading of all kernels on all devices, even with lazy loading enabled.
    // Some tests can create complex dependencies between devices and function loading requires a
    // device synchronization, so loading in the middle of a test can deadlock.
    preloadKernels(deviceCount);

    if (testcasesToRun.size() == 0) {
        // run all testcases
        for (auto testcase : testcases) {
            runTestcase(testcases, testcase->testKey());
        }
    } else {
        for (const auto& testcaseIndex : testcasesToRun) {
            runTestcase(testcases, testcaseIndex);
        }
    }

    output->print();

    for (auto testcase : testcases) {
        delete testcase;
    }

    env->finalize();
    output->printInfo();
    return 0;
}
