cmake_minimum_required(VERSION 3.22)
project(GAiA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-executable_path -Wl,$ORIGIN")

if(WIN32)
  set(LIB_EXT ".dll")
elseif(APPLE)
  set(LIB_EXT ".dylib")
elseif(UNIX)
  set(LIB_EXT ".so")
endif()

set(EP "CPU")

include_directories(src/Stockfish/)
include_directories(src/Stockfish/incbin)
include_directories(src/Stockfish/nnue)
include_directories(src/Stockfish/syzygy)
include_directories(src/Evaluator/)
include_directories(src/Evaluator/SEResNet)
include_directories(include)

set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)
link_directories(${LIB_DIR})

if(${EP} STREQUAL "CPU")
  add_definitions(-DUSE_CPU)
elseif(${EP} STREQUAL "CUDA")
  add_definitions(-DUSE_CUDA)
elseif(${EP} STREQUAL "TENSORRT")
  add_definitions(-DUSE_TENSORRT)
endif()


file (GLOB SERESNET_SOURCES src/Evaluator/SEResNet/*.cpp)
file (GLOB SERESNET_HEADERS src/Evaluator/SEResNet/*.hpp)
add_library(seresnet ${SERESNET_SOURCES} ${SERESNET_HEADERS})
link_libraries(seresnet)

file (GLOB EVALUATOR_SOURCES src/Evaluator/*.cpp)
file (GLOB EVALUATOR_HEADERS src/Evaluator/*.hpp)
add_library(evaluator ${EVALUATOR_SOURCES} ${EVALUATOR_HEADERS})

link_libraries(evaluator)

file (GLOB_RECURSE SRC_FILES src/Stockfish/*.cpp)

add_executable(GAiA ${SRC_FILES})


target_link_libraries(GAiA ${LIB_DIR}/libonnxruntime${LIB_EXT})

if(${EP} STREQUAL "CUDA")
  target_link_libraries(GAiA ${LIB_DIR}/libonnxruntime_providers_tensorrt${LIB_EXT})
elseif(${EP} STREQUAL "TENSORRT")
  target_link_libraries(GAiA ${LIB_DIR}/libonnxruntime_providers_cuda${LIB_EXT})
endif()

if(APPLE)
  add_custom_command(TARGET GAiA 
    POST_BUILD COMMAND 
    ${CMAKE_INSTALL_NAME_TOOL} -add_rpath "@executable_path"
    $<TARGET_FILE:GAiA>)
endif()