cmake_minimum_required(VERSION 3.15)
project(RLMARKET)

set(CMAKE_CXX_STANDARD 14)
SET(BOOST_ROOT "/Users/fangliming/boost")

file(GLOB DATA "src/data/*.cpp")
file(GLOB ENV "src/environment/*.cpp")
file(GLOB EXP "src/experiment/*.cpp")
file(GLOB MARKET "src/market/*.cpp")
file(GLOB UTILITIES "src/utilities/*.cpp")
file(GLOB RL "src/rl/*.cpp")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(RLMARKET main.cpp ${DATA} ${ENV} ${EXP} ${MARKET} ${UTILITIES} ${RL})
target_include_directories(RLMARKET
        PUBLIC
        ./include)
target_link_directories(RLMARKET
        PRIVATE
        ./lib)

target_link_libraries(RLMARKET yaml-cpp pthread spdlog)

# Include/Link dependencies:
find_package(Boost COMPONENTS program_options system filesystem REQUIRED)
if(${Boost_FOUND})
    include_directories(${Boost_INCLUDE_DIRS})

    target_link_libraries(RLMARKET ${Boost_LIBRARIES})
endif(${Boost_FOUND})
