cmake_minimum_required(VERSION 3.15)
project(GraphSeg)

add_subdirectory(pybind11)
pybind11_add_module(GraphSeg GraphSeg.cpp)
