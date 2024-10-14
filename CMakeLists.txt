cmake_minimum_required(VERSION 3.10)
project(sparse_matrix_multiply LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)

add_library(sparse_matrix_multiply MODULE sparse_matrix_multiply.cu)
target_link_libraries(sparse_matrix_multiply PRIVATE pybind11::module)
target_include_directories(sparse_matrix_multiply PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
set_target_properties(sparse_matrix_multiply PROPERTIES PREFIX "${Python3_MODULE_PREFIX}" SUFFIX "${Python3_MODULE_EXTENSION}")