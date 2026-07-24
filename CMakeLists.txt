cmake_minimum_required(VERSION 3.11...3.18)
# The project() name defines the name of the underlying C++ library
project(GTBoW)

# Set the name of the final python module we are building.
# This MUST match the name in the CMakeExtension in setup.py
set(OUTPUT_NAME pyGTBoW)

# -- Tracy profiling (optional) ------------------------------------------------
option(ENABLE_TRACY "Enable Tracy profiling instrumentation" OFF)
if(ENABLE_TRACY)
  include(FetchContent)
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.13.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(tracy)
  # TracyClient is linked into the pyGTBoW shared module; it must be built with -fPIC.
  set_target_properties(TracyClient PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
        add_compile_options(-DUSE_OPENMP)
ENDIF()

# Aggressive release flags for SIMD-accelerated Hamming distance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(ENABLE_NATIVE_ARCH "Optimize for the build machine CPU" OFF)
if(ENABLE_NATIVE_ARCH)
  add_compile_options(-march=native)
endif()

SET(OpenCV_USE_STATIC_LIBS ON)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
link_libraries(
    ${OpenCV_LIBS}
)

# ── FLANN (required by AKMVocabulary) ──────────────────────────────────────────
find_package(flann QUIET)
if(flann_FOUND)
  message(STATUS "Found flann via find_package")
  include_directories(${flann_INCLUDE_DIRS})
  link_libraries(flann::flann)
else()
  # Fallback: look in the conda/pixi prefix
  if(DEFINED ENV{CONDA_PREFIX})
    set(FLANN_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
  elseif(DEFINED ENV{PREFIX})
    set(FLANN_INCLUDE_DIR "$ENV{PREFIX}/include")
  endif()
  if(FLANN_INCLUDE_DIR AND EXISTS "${FLANN_INCLUDE_DIR}/flann/flann.hpp")
    message(STATUS "Found flann headers at ${FLANN_INCLUDE_DIR}")
    include_directories(${FLANN_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Could not find flann/flann.hpp. Install the flann package.")
  endif()
endif()

find_library(LZ4_LIBRARIES NAMES liblz4.so HINTS "lz4/lib")
link_libraries(${LZ4_LIBRARIES})

set (CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(PYTHON_VERSION 3)
message( STATUS "PYTHON_VERSION = ${PYTHON_VERSION}")


# Add cmake_modules to module
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NumPy REQUIRED)


if (UNIX)
  INCLUDE_DIRECTORIES(
      ${OpenCV_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_DIRS}
      ${NUMPY_INCLUDES}
      "${PROJECT_SOURCE_DIR}/src"
      "${PROJECT_SOURCE_DIR}/thirdparty/DBow3/src"
  )
else()
  INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDES}
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/thirdparty/DBow3/src"
  )
endif()


# Find all the original DBoW3 source files
FILE(GLOB DBOW3_SOURCES CONFIGURE_DEPENDS
    "thirdparty/DBow3/src/*.cpp"
)

# Collect all C++ source files from the src directory for our library
FILE(GLOB LIB_SOURCES "src/*.cpp")

# We must exclude the wrapper file from the library, as it's the entry
# point for the Python module itself and contains a main function.
# Including it would cause linker errors.
LIST(FILTER LIB_SOURCES EXCLUDE REGEX ".*gtbow_wrapper\\.cpp$")

# This command now compiles the underlying C++ library ("GTBoW") from
# the files in the src directory.
ADD_LIBRARY(${PROJECT_NAME} STATIC
    ${DBOW3_SOURCES}
    ${LIB_SOURCES}
)

add_subdirectory(thirdparty/pybind11)

include_directories("thirdparty/pybind11/include")
include_directories("thirdparty/pybind11_opencv_numpy")

pybind11_add_module(${OUTPUT_NAME} "src/gtbow_wrapper.cpp" "./thirdparty/pybind11_opencv_numpy/ndarray_converter.cpp")

# This links the python module ("pyGTBoW") to the underlying C++ library ("GTBoW")
target_link_libraries(${OUTPUT_NAME} PRIVATE 
    ${PROJECT_NAME}
)

# Modern OpenMP linkage using target-based approach
if(OPENMP_FOUND)
  target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
  target_link_libraries(${OUTPUT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TRACY)
  target_link_libraries(${PROJECT_NAME} TracyClient)
  target_compile_definitions(${PROJECT_NAME} PUBLIC TRACY_ENABLE)
  target_link_libraries(${OUTPUT_NAME} PRIVATE TracyClient)
endif()
