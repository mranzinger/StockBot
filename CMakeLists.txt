cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(stock_bot)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem date_time)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(stock_price_predictor
               stock_price_predictor.cpp
               hsd_dataset.cpp
               hsd_model.cpp
               hsd_loss.cpp
)
target_link_libraries(stock_price_predictor
                      ${TORCH_LIBRARIES}
                      ${Boost_LIBRARIES})
set_property(TARGET stock_price_predictor PROPERTY CXX_STANDARD 14)