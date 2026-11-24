// rigel-gpu.h -- the CUDA runtime calls of the paged MoE runtime, on NVIDIA (CUDA) or AMD (HIP).
//
// The runtime is written against the CUDA runtime API. With RIGEL_HIP (set by CMake when GGML_HIP is on) the same
// names map to HIP, the way ggml's own HIP build maps them (ggml/src/ggml-cuda/vendors/hip.h).
#pragma once

#if defined(RIGEL_HIP)

#include <hip/hip_runtime.h>

#define cudaError_t                    hipError_t
#define cudaSuccess                    hipSuccess
#define cudaErrorNotReady              hipErrorNotReady
#define cudaGetErrorString             hipGetErrorString
#define cudaGetDevice                  hipGetDevice
#define cudaSetDevice                  hipSetDevice
#define cudaMemGetInfo                 hipMemGetInfo

#define cudaStream_t                   hipStream_t
#define cudaStreamNonBlocking          hipStreamNonBlocking
#define cudaStreamCreateWithFlags      hipStreamCreateWithFlags
#define cudaStreamDestroy              hipStreamDestroy
#define cudaStreamSynchronize          hipStreamSynchronize
#define cudaStreamWaitEvent            hipStreamWaitEvent
#define cudaStreamIsCapturing          hipStreamIsCapturing
#define cudaStreamCaptureStatus        hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone    hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive  hipStreamCaptureStatusActive

#define cudaEvent_t                    hipEvent_t
#define cudaEventCreate                hipEventCreate
#define cudaEventCreateWithFlags       hipEventCreateWithFlags
#define cudaEventDisableTiming         hipEventDisableTiming
#define cudaEventDestroy               hipEventDestroy
#define cudaEventRecord                hipEventRecord
#define cudaEventQuery                 hipEventQuery
#define cudaEventSynchronize           hipEventSynchronize
#define cudaEventElapsedTime           hipEventElapsedTime

#define cudaMalloc                     hipMalloc
#define cudaFree                       hipFree
#define cudaMemset                     hipMemset
#define cudaMemsetAsync                hipMemsetAsync
#define cudaMemcpy                     hipMemcpy
#define cudaMemcpyAsync                hipMemcpyAsync
#define cudaMemcpy2DAsync              hipMemcpy2DAsync
#define cudaMemcpyHostToDevice         hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost         hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice       hipMemcpyDeviceToDevice

// Pinned host memory. Memory the GPU reads while the host writes it (the doorbells) must be coherent on AMD, so
// mapped allocations are also made coherent (fine-grained) rather than left to the default.
#define cudaHostAlloc(p, n, f)         hipHostMalloc((p), (n), (f))
#define cudaFreeHost                   hipHostFree
#define cudaHostAllocDefault           hipHostMallocDefault
#define cudaHostAllocPortable          hipHostMallocPortable
#define cudaHostAllocMapped            (hipHostMallocMapped | hipHostMallocCoherent)
#define cudaHostAllocWriteCombined     hipHostMallocWriteCombined
#define cudaHostGetDevicePointer       hipHostGetDevicePointer
#define cudaHostRegister               hipHostRegister
#define cudaHostRegisterDefault        hipHostRegisterDefault
#define cudaHostUnregister             hipHostUnregister

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// Warp shuffles: HIP's *_sync forms want a 64-bit mask; the runtime always passes a full 32-lane mask.
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...)  __shfl_xor(__VA_ARGS__)
#define __shfl_sync(mask, ...)      __shfl(__VA_ARGS__)
#endif

#else

#include <cuda_runtime.h>

#endif
