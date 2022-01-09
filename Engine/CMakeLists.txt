cmake_minimum_required(VERSION 3.17)
project(GAiA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -mavx2")

include_directories(src/Stockfish/)
include_directories(src/Stockfish/incbin)
include_directories(src/Stockfish/nnue)
include_directories(src/Stockfish/syzygy)
include_directories(src/Evaluator/)
include_directories(src/Evaluator/SEResNet)
include_directories(include)


file (GLOB SERESNET_SOURCES src/Evaluator/SEResNet/*.cpp)
file (GLOB SERESNET_HEADERS src/Evaluator/SEResNet/*.hpp)
add_library(seresnet ${SERESNET_SOURCES} ${SERESNET_HEADERS})
link_libraries(seresnet)

file (GLOB EVALUATOR_SOURCES src/Evaluator/*.cpp)
file (GLOB EVALUATOR_HEADERS src/Evaluator/*.hpp)
add_library(evaluator ${EVALUATOR_SOURCES} ${EVALUATOR_HEADERS})

link_libraries(seresnet evaluator)

add_executable(GAiA
  src/Stockfish/main.cpp
  src/Stockfish/benchmark.cpp
  src/Stockfish/bitbase.cpp
  src/Stockfish/bitboard.cpp
  src/Stockfish/endgame.cpp
  src/Stockfish/evaluate.cpp
  src/Stockfish/material.cpp
  src/Stockfish/misc.cpp
  src/Stockfish/movegen.cpp
  src/Stockfish/movepick.cpp
  src/Stockfish/pawns.cpp
  src/Stockfish/position.cpp
  src/Stockfish/psqt.cpp
  src/Stockfish/search.cpp
  src/Stockfish/thread.cpp
  src/Stockfish/timeman.cpp
  src/Stockfish/tt.cpp
  src/Stockfish/uci.cpp
  src/Stockfish/ucioption.cpp
  src/Stockfish/tune.cpp
  src/Stockfish/syzygy/tbprobe.cpp
  src/Stockfish/nnue/evaluate_nnue.cpp
  src/Stockfish/nnue/features/half_kp.cpp
  )

target_link_libraries(GAiA ${CMAKE_CURRENT_LIST_DIR}/lib/libonnxruntime.dylib)