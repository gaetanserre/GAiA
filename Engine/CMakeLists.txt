cmake_minimum_required(VERSION 3.17)
project(Deep_ViCTORIA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib/)

include_directories(src/Stockfish/incbin)
include_directories(src/Stockfish/nnue)
include_directories(src/Stockfish/syzygy)



add_executable(Deep_ViCTORIA
		src/Stockfish/benchmark.cpp
		src/Stockfish/bitbase.cpp
		src/Stockfish/bitboard.cpp
		src/Stockfish/endgame.cpp
		src/Stockfish/evaluate.cpp
		src/Stockfish/main.cpp
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
		src/Evaluator/evaluator.cpp)

target_include_directories(Deep_ViCTORIA PRIVATE src/Evaluator/cppflow $ENV{HOME}/libtensorflow/include/)
target_link_libraries (Deep_ViCTORIA "${TENSORFLOW_LIB}")