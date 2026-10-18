cmake_minimum_required(VERSION 3.18)
project(loupe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch: unzip the CPU build into third_party/libtorch (scripts/get-deps.sh),
# or point CMAKE_PREFIX_PATH at another copy.
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)

add_library(loupe_core STATIC src/loupe.cpp)
target_include_directories(loupe_core PUBLIC src)
target_link_libraries(loupe_core PUBLIC ${TORCH_LIBRARIES})

add_executable(experiment src/experiment.cpp)
target_link_libraries(experiment PRIVATE loupe_core)

add_executable(serve src/serve.cpp)
target_include_directories(serve PRIVATE third_party)
target_link_libraries(serve PRIVATE loupe_core)

enable_testing()
add_executable(tests tests/tests.cpp)
target_link_libraries(tests PRIVATE loupe_core)
add_test(NAME tests COMMAND tests)
