llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    moe-cpu.cpp
    moe_pager.cpp
    rigel-port.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

# The paged MoE runtime (moe_pager, the doorbell and the predictor) runs on a GPU: CUDA (NVIDIA, Linux) as upstream, or HIP (AMD) in Rigel. rigel-port.h supplies io_uring and the other POSIX pieces on Windows.
if (GGML_CUDA)
    enable_language(CUDA)
    target_sources(llama PRIVATE moe_predictor.cu moe_doorbell.cu)
    set_target_properties(llama PROPERTIES CUDA_ARCHITECTURES 86)
    set_source_files_properties(moe_predictor.cu moe_doorbell.cu PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    target_include_directories(llama PRIVATE ${_cmake_CUDAToolkit_include_directories})
    target_link_directories(llama PRIVATE ${_cmake_CUDAToolkit_library_directories})
    if (NOT WIN32)
        target_link_libraries(llama PRIVATE uring)
    endif()
elseif (GGML_HIP)
    find_package(hip REQUIRED)
    if (WIN32)
        # CMake has no HIP language on Windows: like ggml-hip, compile the kernels as C++ with hip::device (which
        # adds -x hip and the GPU_TARGETS offload archs). They get a library of their own so that only they, not
        # every source of llama, are compiled for the GPU.
        add_library(rigel-hip-kernels STATIC moe_predictor.cu moe_doorbell.cu)
        set_source_files_properties(moe_predictor.cu moe_doorbell.cu PROPERTIES LANGUAGE CXX)
        target_include_directories(rigel-hip-kernels PRIVATE . ../include)
        target_compile_features(rigel-hip-kernels PRIVATE cxx_std_17)
        target_compile_definitions(rigel-hip-kernels PRIVATE RIGEL_HIP)
        target_link_libraries(rigel-hip-kernels PRIVATE ggml hip::device)
        target_link_libraries(llama PRIVATE rigel-hip-kernels)
    else()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
        endif()
        enable_language(HIP)
        target_sources(llama PRIVATE moe_predictor.cu moe_doorbell.cu)
        set_source_files_properties(moe_predictor.cu moe_doorbell.cu PROPERTIES LANGUAGE HIP SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    target_compile_definitions(llama PRIVATE RIGEL_HIP)
    target_link_libraries(llama PRIVATE hip::host)
    if (NOT WIN32)
        target_link_libraries(llama PRIVATE uring)
    endif()
else()
    message(FATAL_ERROR "Rigel: the paged MoE runtime needs a GPU backend: GGML_CUDA=ON or GGML_HIP=ON")
endif()

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

# Rigel: a check that the expert pager reads exactly the GGUF's bytes on Windows (tests/test_rigel_pager.cpp).
option(RIGEL_BUILD_PAGER_TEST "Rigel: build test-rigel-pager" OFF)
if (RIGEL_BUILD_PAGER_TEST AND GGML_HIP)
    add_executable(test-rigel-pager ../tests/test_rigel_pager.cpp)
    target_include_directories(test-rigel-pager PRIVATE .)
    target_compile_definitions(test-rigel-pager PRIVATE RIGEL_HIP)
    target_link_libraries(test-rigel-pager PRIVATE llama ggml hip::host)
endif()

# Rigel: debug info for the llama library and the server (a crash report can then be turned into source lines).
option(RIGEL_DEBUG_INFO "Rigel: build llama with debug info" OFF)
if (RIGEL_DEBUG_INFO)
    target_compile_options(llama PRIVATE -g -gcodeview)
    target_link_options(llama INTERFACE -g)
endif()
