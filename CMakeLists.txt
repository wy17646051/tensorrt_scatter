cmake_minimum_required(VERSION 3.10.0)
project(tensorrtscatter VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)


find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

if (NOT DEFINED TENSORRT_PREFIX_PATH)
    message(FATAL_ERROR "TENSORRT_PREFIX_PATH is not defined")
endif()
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_PREFIX_PATH}/include")
file(GLOB TENSORRT_LIBRARIES "${TENSORRT_PREFIX_PATH}/lib/*.so")

set(HEADER_DIRS "include")
file(GLOB SOURCES 
    "src/*.cpp" "src/*.cu" 
    "src/plugin/*.cpp" "src/plugin/*.cu")


add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_include_directories(${PROJECT_NAME} 
    PUBLIC ${HEADER_DIRS} 
    PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ${HEADER_DIRS}/common
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES}
)
