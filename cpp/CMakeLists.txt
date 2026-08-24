cmake_minimum_required(VERSION 3.14)
project(AegisSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimized performance compilation flags
if(MSVC)
    add_compile_options(/O2 /W4 /permissive-)
else()
    add_compile_options(-O3 -Wall -Wextra -pedantic)
endif()

add_executable(simulator simulator.cpp)

# Cross-platform library linkage
if(WIN32)
    target_link_libraries(simulator PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(simulator PRIVATE Threads::Threads)
endif()
