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

# python site-packages folder
execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

# it _must_ go before PROJECT(PyOpenCV) in order to work
set(CMAKE_INSTALL_PREFIX "${PYTHON_SITE_PACKAGES}/pyopencv" CACHE PATH "PyOpenCV's installation path")

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
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
if(NOT PYTHONLIBS_FOUND OR NOT PYTHON_EXECUTABLE)
message(SEND_ERROR "You need Python to build a Python extension")
endif(NOT PYTHONLIBS_FOUND OR NOT PYTHON_EXECUTABLE)

# cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
    set(PYTHONLIBS_FOUND ON)
endif(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)

# python version
execute_process(COMMAND ${PYTHON_EXECUTABLE} --version ERROR_VARIABLE 
    PYTHON_VERSION_FULL OUTPUT_STRIP_TRAILING_WHITESPACE)
    
# The code below prints the Python extension for the current system
FILE(WRITE "${CMAKE_BINARY_DIR}/getmodsuffix.py"
"import imp
for s in imp.get_suffixes():
    if s[1] == 'rb' and s[0][0] == '.':
        break
print s[0],
")
# Now execute it and remove any newlines from output
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/getmodsuffix.py
    OUTPUT_VARIABLE _modsuffix)
STRING(REPLACE "\n" "" _modsuffix ${_modsuffix}) # Setup specific properties of the Python target


set(Boost_USE_STATIC_LIBS OFF) # need dynamic linking
set(Boost_python_DYN_LINK ON)
find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)

include_directories(
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/package
	${PROJECT_SOURCE_DIR}/package/extras
	${PROJECT_SOURCE_DIR}/package/extras/core
	${PROJECT_SOURCE_DIR}/package/extras/sdopencv
	${PROJECT_SOURCE_DIR}/package/extras/numpy_include
	${PROJECT_SOURCE_DIR}/package/stl_ext
	${PYTHON_INCLUDE_PATH}
)

# 'extras' library
file(GLOB_RECURSE EXTRA_SRCS package/extras/*)
add_library(pyopencv_extras SHARED ${EXTRA_SRCS})
target_link_libraries(pyopencv_extras ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
install(TARGETS pyopencv_extras DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT main)

# 'stl' module
file(GLOB STL_SRCS package/stl_ext/*)
add_library(stl_ext SHARED ${STL_SRCS})
target_link_libraries(stl_ext pyopencv_extras ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
set_target_properties(stl_ext PROPERTIES 
    PREFIX "" # There is no prefix even on UNIXes
    SUFFIX "${_modsuffix}") # The extension got from Python libraries
install(TARGETS stl_ext DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT main)

file(GLOB PY_SRCS package/*.py)
install(FILES ${PY_SRCS} DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT main)

include(CPack)
