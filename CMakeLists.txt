cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(stock_bot)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem date_time)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(stock_price_predictor
               stock_price_predictor.cpp
               hsd_dataset.cpp
)
target_link_libraries(stock_price_predictor
                      ${TORCH_LIBRARIES}
                      ${Boost_LIBRARIES})
set_property(TARGET stock_price_predictor PROPERTY CXX_STANDARD 14)