cmake_minimum_required(VERSION 3.15)
project(motion_correction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(KIMODO_ARM64 FALSE)
set(KIMODO_UNIVERSAL2 FALSE)
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    # Respect an explicit target architecture when cross-compiling on a Mac.
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(KIMODO_ARM64 TRUE)
    endif()
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        set(KIMODO_UNIVERSAL2 TRUE)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    set(KIMODO_ARM64 TRUE)
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find or fetch pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# MotionCorrection uses x86 SIMD intrinsics. On ARM64, SIMDe maps those
# intrinsics to NEON while preserving the existing, well-tested math code.
if(KIMODO_ARM64)
    find_path(
        SIMDE_INCLUDE_DIR
        NAMES simde/x86/avx.h
        HINTS /opt/homebrew/include /usr/local/include
    )
    if(NOT SIMDE_INCLUDE_DIR)
        message(STATUS "SIMDe not found, fetching v0.8.2 from GitHub...")
        FetchContent_Declare(
            simde
            GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
            GIT_TAG v0.8.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(simde)
        set(SIMDE_INCLUDE_DIR "${simde_SOURCE_DIR}")
    endif()
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

if(KIMODO_ARM64)
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
    target_compile_options(motion_correction_cpp_base PRIVATE /W4)
    if(NOT KIMODO_ARM64)
        target_compile_options(motion_correction_cpp_base PRIVATE /arch:AVX)
    endif()
elseif(KIMODO_UNIVERSAL2)
    # Preserve AVX/SSE acceleration in the x86_64 slice while the arm64 slice
    # uses SIMDe/NEON. Plain -mavx would incorrectly apply to both slices.
    target_compile_options(
        motion_correction_cpp_base PRIVATE
        -Wall -Wextra
        -Xarch_x86_64 -msse4.1
        -Xarch_x86_64 -mavx
    )
elseif(KIMODO_ARM64)
    # Apple Silicon and other ARM64 targets use SIMDe/NEON, so x86 ISA flags
    # such as -mavx and -msse4.1 must not be passed to the compiler.
    target_compile_options(motion_correction_cpp_base PRIVATE -Wall -Wextra)
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
