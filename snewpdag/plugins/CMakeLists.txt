cmake_minimum_required(VERSION 3.15)

project(RecursionIntegral LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_library(_recursion_integral MODULE RecursionIntegral.cc)
target_compile_features(_recursion_integral PRIVATE cxx_std_17)
target_compile_options(_recursion_integral PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_link_libraries(_recursion_integral PRIVATE Python::Module)

# Keep the binary in the plugin package, mirroring Moore's separate compiled
# component and Python-configuration layers.
set_target_properties(_recursion_integral PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
