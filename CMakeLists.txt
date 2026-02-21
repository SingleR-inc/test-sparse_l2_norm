cmake_minimum_required(VERSION 3.14)

project(singler-perf
    VERSION 1.0.0
    DESCRIPTION "Performance tests for SingleR's scaling calculations"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG v2.6.1
)

FetchContent_Declare(
    eztimer
    GIT_REPOSITORY https://github.com/tatami-inc/eztimer
    GIT_TAG master
)

FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(eztimer)

add_executable(basic basic.cpp)
target_link_libraries(basic CLI11::CLI11 tatami::eztimer)

add_executable(fine_tune fine_tune.cpp)
target_link_libraries(fine_tune CLI11::CLI11 tatami::eztimer)
