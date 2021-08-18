cmake_minimum_required(VERSION 3.17)
project(Deep_ViCTORIA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

include_directories(src/Stockfish/)
include_directories(src/Stockfish/incbin)
include_directories(src/Stockfish/nnue)
include_directories(src/Stockfish/syzygy)
include_directories(src/Evaluator/)
include_directories(src/Evaluator/NeuralNet)

file (GLOB NEURALNET_SOURCES src/Evaluator/NeuralNet/*.cpp)
file (GLOB NEURALNET_HEADERS src/Evaluator/NeuralNet/*.hpp)
add_library(neuralnet ${NEURALNET_SOURCES} ${NEURALNET_HEADERS})

link_libraries(neuralnet)

file (GLOB EVALUATOR_SOURCES src/Evaluator/*.cpp)
file (GLOB EVALUATOR_HEADERS src/Evaluator/*.hpp)
add_library(evaluator ${EVALUATOR_SOURCES} ${EVALUATOR_HEADERS})

link_libraries(neuralnet evaluator)


add_executable(Deep_ViCTORIA
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
