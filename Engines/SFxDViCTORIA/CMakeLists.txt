cmake_minimum_required(VERSION 3.17)
project(Deep_ViCTORIA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib/)

include_directories(src/incbin)
include_directories(src/nnue)
include_directories(src/syzygy)



add_executable(SFxDViCTORIA
    src/benchmark.cpp
    src/bitbase.cpp
    src/bitboard.cpp
    src/endgame.cpp
    src/evaluate.cpp
    src/main.cpp
	src/material.cpp
    src/misc.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/pawns.cpp
    src/position.cpp
    src/psqt.cpp
	src/search.cpp
    src/thread.cpp
    src/timeman.cpp
    src/tt.cpp
    src/uci.cpp
    src/ucioption.cpp
    src/tune.cpp
    src/syzygy/tbprobe.cpp
	src/nnue/evaluate_nnue.cpp
    src/nnue/features/half_kp.cpp
)

target_include_directories(SFxDViCTORIA PRIVATE src/cppflow $ENV{HOME}/libtensorflow/include/)
target_link_libraries (SFxDViCTORIA "${TENSORFLOW_LIB}")