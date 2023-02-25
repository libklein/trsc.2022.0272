cmake_minimum_required(VERSION 3.15)

# To force building for a specific python version, set Python_ROOT_DIR to the respective path.
# See the FIND_PACKAGE python cmake documentation for details.
FIND_PACKAGE(Python COMPONENTS Interpreter Development REQUIRED)

verbose_message("Found python binary: ${Python_EXECUTABLE} (Version ${Python_VERSION})")

set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_LIBS ${Python_LIBS})
FIND_PACKAGE(pybind11 CONFIG REQUIRED)

project(evspnl
        LANGUAGES CXX)

set(binding_src_files src/bindings.cpp
        src/pwl.cpp
        src/charger.cpp
        src/tour.cpp
        src/network.cpp
        src/solver.cpp
        src/label.cpp
        src/instance.cpp
        src/battery.cpp
        src/cost_profile.cpp
        src/util.cpp
        )

pybind11_add_module(evspnl ${binding_src_files})

#target_compile_options(frvcp_bindings PUBLIC "-flto")
#target_link_options(frvcp_bindings PUBLIC "-flto")

target_include_directories(evspnl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(evspnl PRIVATE frvcp)
target_include_directories(frvcp PRIVATE ${Python_INCLUDE_DIRS})

#
# Set the compiler standard
#
target_compile_features(evspnl PUBLIC cxx_std_20)
