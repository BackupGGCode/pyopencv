# ----------------------------------------------------------------------------
#  Root CMake file for PyOpenCV
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_OPENCV_ROOT>
#
#
#   - JUN-2010: Initial version, by Dat Chu
#   - JUL-2010: Revision 1, by Minh-Tri Pham
#
# ----------------------------------------------------------------------------

project(pyopencv)
cmake_minimum_required(VERSION 2.4)

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# Don't try the Debug mode, it's too long to compile.
SET(CMAKE_BUILD_TYPE "Release")

# Need Python
# set(Boost_USE_STATIC_LIBS OFF)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
if(NOT PYTHONLIBS_FOUND OR NOT PYTHON_EXECUTABLE)
message(SEND_ERROR "You need Python to build a Python extension")
endif(NOT PYTHONLIBS_FOUND OR NOT PYTHON_EXECUTABLE)

# cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
    set(PYTHONLIBS_FOUND ON)
endif(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)

execute_process(COMMAND ${PYTHON_EXECUTABLE} --version ERROR_VARIABLE 
    PYTHON_VERSION_FULL OUTPUT_STRIP_TRAILING_WHITESPACE)
    
execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)

file(GLOB_RECURSE SRCS pyopencv/pyopencvext/*)

include_directories(
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/pyopencv
	${PROJECT_SOURCE_DIR}/pyopencv/pyopencvext
	${PROJECT_SOURCE_DIR}/pyopencv/pyopencvext/core
	${PROJECT_SOURCE_DIR}/pyopencv/pyopencvext/sdopencv
	${PROJECT_SOURCE_DIR}/pyopencv/pyopencvext/numpy_include
	${PYTHON_INCLUDE_PATH}
)

add_library(pyopencvext SHARED ${SRCS})

target_link_libraries(pyopencvext ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
