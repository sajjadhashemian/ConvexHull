cmake_minimum_required(VERSION 3.18)
project(approxhull_cpp LANGUAGES CXX)

include(GNUInstallDirs)

find_package(pybind11 CONFIG REQUIRED)

option(APPROXHULL_OPENMP "Enable OpenMP" ON)

add_library(approxhull_cpp MODULE
    src/argmax_dot.cpp
    src/householder.cpp
    bindings/pybind_module.cpp
)

target_include_directories(approxhull_cpp PRIVATE include)

target_compile_features(approxhull_cpp PRIVATE cxx_std_17)

if (APPROXHULL_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(approxhull_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(approxhull_cpp PRIVATE APPROXHULL_OPENMP=1)
  endif()
endif()

set_target_properties(approxhull_cpp PROPERTIES
  PREFIX ""
  OUTPUT_NAME "approxhull_cpp"
)

pybind11_extension(approxhull_cpp)

install(TARGETS approxhull_cpp LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
