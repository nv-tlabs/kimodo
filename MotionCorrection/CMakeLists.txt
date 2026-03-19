cmake_minimum_required(VERSION 3.15)
project(motion_correction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find brew package
find_path(SIMDE_INCLUDE_DIR
    NAMES simde/x86/sse.h
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)")
    if(NOT SIMDE_INCLUDE_DIR)
        message(FATAL_ERROR "SIMDe headers not found. Please install simde or set CMAKE_PREFIX_PATH.")
    endif()
endif()

# Find or fetch pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find or fetch Eigen
find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

# Source files
set(MATH_SOURCES
    src/cpp/Math/Matrix.cpp
    src/cpp/Math/Quaternion.cpp
    src/cpp/Math/Transform.cpp
    src/cpp/Math/Types.cpp
    src/cpp/Math/Vector.cpp
)

set(ANIM_SOURCES
    src/cpp/AnimProcessing/InverseKinematics.cpp
    src/cpp/AnimProcessing/TrajectoryCorrector.cpp
    src/cpp/AnimProcessing/Utility.cpp
)

# Create static library for the core functionality
add_library(motion_correction_cpp_base STATIC ${MATH_SOURCES} ${ANIM_SOURCES})

# Enable Position Independent Code (required for linking into shared library)
set_target_properties(motion_correction_cpp_base PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(motion_correction_cpp_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

if(SIMDE_INCLUDE_DIR)
    target_include_directories(motion_correction_cpp_base PUBLIC ${SIMDE_INCLUDE_DIR})
endif()

if(TARGET Eigen3::Eigen)
    target_link_libraries(motion_correction_cpp_base PUBLIC Eigen3::Eigen)
else()
    target_link_libraries(motion_correction_cpp_base PUBLIC eigen)
endif()

target_compile_definitions(motion_correction_cpp_base PUBLIC EIGEN_MPL2_ONLY)

# Compiler-specific settings
if(MSVC)
    # MSVC-specific flags
    target_compile_options(motion_correction_cpp_base PRIVATE /W4 /arch:AVX)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    # ARM64/NEON flags
	target_compile_options(motion_correction_cpp_base PRIVATE -Wall -Wextra -O3)

else()
    # GCC/Clang flags (also applies to MinGW on Windows)
    # Enable SSE4.1 and AVX instructions for SIMD operations
    target_compile_options(motion_correction_cpp_base PRIVATE -Wall -Wextra -msse4.1 -mavx)
endif()

# Python bindings
pybind11_add_module(_motion_correction src/cpp/BindingsPython.cpp)

target_link_libraries(_motion_correction PRIVATE motion_correction_cpp_base)

target_include_directories(_motion_correction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)


# Install the Python module
install(TARGETS _motion_correction LIBRARY DESTINATION python/motion_correction)
install(FILES python/motion_correction/__init__.py DESTINATION python/motion_correction)
install(FILES python/motion_correction/motion_postprocess.py DESTINATION python/motion_correction)
