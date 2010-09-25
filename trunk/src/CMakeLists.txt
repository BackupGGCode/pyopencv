# ----------------------------------------------------------------------------
#  Root CMake file for PyOpenCV
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_PYOPENCV_ROOT>
#
#
#   - JUN-2010: Initial version, by Dat Chu
#   - JUL-2010: Adapted, by Minh-Tri Pham, from OpenCV's CMakeLists.txt -- thanks WG people
#   - AUG-2010: Fixed linking bugs on Mac, by Attila
#
# ----------------------------------------------------------------------------

project(pyopencv)
cmake_minimum_required(VERSION 2.4)

# --------------------------------------------------------------
# Detect Python at set up the installation path:
# --------------------------------------------------------------
# Need Python
find_package(PythonInterp REQUIRED)

# python site-packages folder
execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

# it _must_ go before PROJECT(PyOpenCV) in order to work
set(PYOPENCV_INSTALL_PATH ${PYTHON_SITE_PACKAGES}/pyopencv CACHE PATH "PyOpenCV's installation path")

# Need Python-dev
find_package(PythonLibs REQUIRED)
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

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# ----------------------------------------------------------------------------
#  Current version number:
# ----------------------------------------------------------------------------
set(PYOPENCV_VERSION_MAJOR "2.1.0.wr1")
set(PYOPENCV_VERSION_MINOR "2")
set(PYOPENCV_VERSION_PATCH "0")
set(PYOPENCV_VERSION "${PYOPENCV_VERSION_MAJOR}.${PYOPENCV_VERSION_MINOR}.${PYOPENCV_VERSION_PATCH}")

# ----------------------------------------------------------------------------
#  Variables for cvconfig.h.cmake
# ----------------------------------------------------------------------------
set(PACKAGE "pyopencv")
set(PACKAGE_BUGREPORT "pmtri80@gmail.com")
set(PACKAGE_NAME "pyopencv")
set(PACKAGE_STRING "${PACKAGE} ${PYOPENCV_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE}")
set(PACKAGE_VERSION "${PYOPENCV_VERSION}")

# ----------------------------------------------------------------------------
#  Autodetect if we are in a SVN repository
# ----------------------------------------------------------------------------
find_program(SVNVERSION_PATH svnversion)
mark_as_advanced(force SVNVERSION_PATH)
if(SVNVERSION_PATH)
    message(STATUS "Extracting svn version, please wait...")
    execute_process(COMMAND ${SVNVERSION_PATH} -n ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE SVNVERSION_RESULT)

    if(SVNVERSION_RESULT MATCHES "exported")
       # This is NOT a svn repository:
       set(PYOPENCV_SVNVERSION "")
    else()
        set(PYOPENCV_SVNVERSION " svn:${SVNVERSION_RESULT}")
    endif()

    message(STATUS "SVNVERSION: ${PYOPENCV_SVNVERSION}")
else()
    # We don't have svnversion:
    set(PYOPENCV_SVNVERSION "")
endif()


# ----------------------------------------------------------------------------
# Detect GNU version:
# ----------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE CMAKE_PYOPENCV_GCC_VERSION_FULL
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Typical output in CMAKE_PYOPENCV_GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
    #  Look for the version number
    string(REGEX MATCH "[0-9].[0-9].[0-9]" CMAKE_GCC_REGEX_VERSION "${CMAKE_PYOPENCV_GCC_VERSION_FULL}")

    # Split the three parts:
    string(REGEX MATCHALL "[0-9]" CMAKE_PYOPENCV_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

    list(GET CMAKE_PYOPENCV_GCC_VERSIONS 0 CMAKE_PYOPENCV_GCC_VERSION_MAJOR)
    list(GET CMAKE_PYOPENCV_GCC_VERSIONS 1 CMAKE_PYOPENCV_GCC_VERSION_MINOR)
    
    set(CMAKE_PYOPENCV_GCC_VERSION ${CMAKE_PYOPENCV_GCC_VERSION_MAJOR}${CMAKE_PYOPENCV_GCC_VERSION_MINOR})
    math(EXPR CMAKE_PYOPENCV_GCC_VERSION_NUM "${CMAKE_PYOPENCV_GCC_VERSION_MAJOR}*100 + ${CMAKE_PYOPENCV_GCC_VERSION_MINOR}")
    message(STATUS "Detected version of GNU GCC: ${CMAKE_PYOPENCV_GCC_VERSION} (${CMAKE_PYOPENCV_GCC_VERSION_NUM})")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*)
        # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
    endif()
    
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(X86_64 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(X86_64 1)
    endif()
    
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        set(X86 1)
    endif()
        
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()
    
    if(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
        # disable SSE stuff to avoid any runtime segfault issues -- by Minh-Tri
        # set(ENABLE_SSE ON CACHE BOOL "Enable SSE for GCC")
        # set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        # set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
        # set(ENABLE_SSSE3 OFF CACHE BOOL "Enable SSSE3 for GCC")
        #set(ENABLE_SSE4_1 OFF CACHE BOOL "Enable SSE4.1 for GCC")
    endif()
endif()



# ----------------------------------------------------------------------------
#       CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------

# Boost.Python
set(Boost_USE_STATIC_LIBS OFF) # need dynamic linking
set(Boost_Python_DYN_LINK ON)
find_package(Boost COMPONENTS python REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)

# include_dirs
set(PYOPENCV_INCLUDE_DIRS
	${PYTHON_INCLUDE_PATH}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/package
	${PROJECT_SOURCE_DIR}/package/extras
	${PROJECT_SOURCE_DIR}/package/extras/core
	${PROJECT_SOURCE_DIR}/package/extras/sdopencv
)

# library_dirs
set(PYOPENCV_LIBRARY_DIRS
    ${Boost_LIBRARY_DIRS}
    ${OpenCV_LIB_DIR}
)

# libraries
set(PYOPENCV_LIBRARIES ${Boost_PYTHON_LIBRARY_RELEASE})
set(FOUND_LIB 0)
foreach(LIB ${OpenCV_LIBS})
    if(LIB STREQUAL "optimized")
        set(FOUND_LIB 1)
    else()
        if(FOUND_LIB EQUAL 1)
            set(PYOPENCV_LIBRARIES ${LIB} ${PYOPENCV_LIBRARIES})
            set(FOUND_LIB 0)
        endif()
    endif()
endforeach()

# install files
set(PYOPENCV_INSTALL_FILES "")

# Detect NumPy include dirs. If not found, use our copy of NumPy 1.4.0 include dir
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
    RESULT_VARIABLE PYTHON_NUMPY_PROCESS
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(PYTHON_NUMPY_PROCESS EQUAL 0)
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIRS}/numpy)
    message(STATUS "NumPy include directory detected at: ${PYTHON_NUMPY_INCLUDE_DIRS}")
else()
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/package/extras/numpy_include)
    message(STATUS "NumPy include directory not detected, using our copy of numpy 1.4.1 at: ${PYTHON_NUMPY_INCLUDE_DIRS}")
endif()
set(PYOPENCV_INCLUDE_DIRS ${PYOPENCV_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIRS})

# ----------------------------------------------------------------------------
#           Set the maximum level of warnings:
# ----------------------------------------------------------------------------
# Should be set to true for development
set(PYOPENCV_WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")

set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")

if(MSVC)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
    # 64-bit portability warnings, in MSVC8
    if(MSVC80)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Wp64")
    endif()
    #if(MSVC90)
    #    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _BIND_TO_CURRENT_CRT_VERSION=1 /D _BIND_TO_CURRENT_VCLIBS_VERSION=1")
    #endif()

    set(EXTRA_EXE_LINKER_FLAGS_RELEASE "${EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")

    # Remove unreferenced functions: function level linking
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Gy")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Zi")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)

    if(WIN32)
        if(CMAKE_CXX_COMPILER MATCHES "64")
            set(MINGW64 1)
        endif()
    endif()

    # High level of warnings.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    if(PYOPENCV_WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif()

    if(X86)
        if(NOT X86_64)
            if(NOT APPLE)
                set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
            endif()
        endif()
    endif()

    # Other optimizations
    if(USE_O2)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    endif()
    if(USE_FAST_MATH)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(ENABLE_POWERPC)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(ENABLE_SSE)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse")
    endif()
    if(ENABLE_SSE2)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse2")
    endif()
    # SSE3 and further should be disabled under MingW because it generates compiler errors
    if(NOT MINGW)
       if(ENABLE_SSE3)
          set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse3")
       endif()
       if(${CMAKE_PYOPENCV_GCC_VERSION_NUM} GREATER 402)
          set(HAVE_GCC43_OR_NEWER 1)
       endif()
       if(HAVE_GCC43_OR_NEWER OR APPLE)
          if(ENABLE_SSSE3)
             set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
          endif()
          #if(ENABLE_SSE4_1)
          #   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
          #endif()
       endif()
    endif()
    
    if(X86 OR X86_64)
        if(NOT APPLE)
            if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
            endif()
        endif()
    endif()

    # Profiling?
    if(ENABLE_PROFILING)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
    else()
        # Remove unreferenced functions: function level linking
        if(NOT APPLE)
            set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
        endif()
    endif()

    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -DDEBUG -D_DEBUG")
endif()

# Add user supplied extra options (optimization, etc...)
# ==========================================================
set(PYOPENCV_EXTRA_C_FLAGS "" CACHE STRING "Extra compiler options")
set(PYOPENCV_EXTRA_C_FLAGS_RELEASE "" CACHE STRING "Extra compiler options for Release build")
set(PYOPENCV_EXTRA_C_FLAGS_DEBUG "" CACHE STRING "Extra compiler options for Debug build")
set(PYOPENCV_EXTRA_EXE_LINKER_FLAGS "" CACHE STRING "Extra linker flags")
set(PYOPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "Extra linker flags for Release build")
set(PYOPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Extra linker flags for Debug build")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${PYOPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} ${PYOPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${PYOPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${PYOPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${PYOPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${PYOPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS} ${PYOPENCV_EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE} ${PYOPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG} ${PYOPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}")

# Use the Release build type.
SET(CMAKE_BUILD_TYPE Release)

if (WIN32 AND MSVC)
    # avoid warnings from MSVC about overriding the /W* option
    # we replace /W3 with /W4 only for C++ files,
    # since all the 3rd-party libraries OpenCV uses are in C,
    # and we do not care about their warnings.
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    # allow extern "C" functions throw exceptions
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if("${CMAKE_CONFIGURE_LDFLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_CONFIGURE_LDFLAGS}")
endif("${CMAKE_CONFIGURE_LDFLAGS}")



# ----------------------------------------------------------------------------
#                       REAL PYOPENCV STUFF
# ----------------------------------------------------------------------------

set(PYTHON_MODULE_LIST)

# include_directories
include_directories(${PYOPENCV_INCLUDE_DIRS})

# ---------------------------------------------
# write to 'package/pyopencv/config.py'
# ---------------------------------------------
file(WRITE package/pyopencv/config.py "
PYOPENCV_VERSION_MAJOR = \"${PYOPENCV_VERSION_MAJOR}\" 
PYOPENCV_VERSION_MINOR = \"${PYOPENCV_VERSION_MINOR}\"
PYOPENCV_VERSION_PATCH = \"${PYOPENCV_VERSION_PATCH}\"
PYOPENCV_VERSION = \"${PYOPENCV_VERSION}\"

")

# ---------------------------------------------
# write to 'config.py'
# ---------------------------------------------

# version
file(WRITE config.py "from setuptools import Extension, Library
import os.path as op
from glob import glob

PYOPENCV_VERSION_MAJOR = \"${PYOPENCV_VERSION_MAJOR}\" 
PYOPENCV_VERSION_MINOR = \"${PYOPENCV_VERSION_MINOR}\"
PYOPENCV_VERSION_PATCH = \"${PYOPENCV_VERSION_PATCH}\"
PYOPENCV_VERSION = \"${PYOPENCV_VERSION}\"

")

# flags
file(APPEND config.py "
CXX_FLAGS = '${CMAKE_CXX_FLAGS_RELEASE}'.split()
LINK_FLAGS = '${CMAKE_EXE_LINKER_FLAGS_RELEASE}'.split()

")

# include dirs
file(APPEND config.py "include_dirs = [")
foreach(DIR ${PYOPENCV_INCLUDE_DIRS})
    string(REPLACE "\\" "/" DIR ${DIR})
    file(APPEND config.py "\"${DIR}\", ")
endforeach()
file(APPEND config.py "]
")

# library dirs
file(APPEND config.py "library_dirs = [")
foreach(DIR ${PYOPENCV_LIBRARY_DIRS})
    file(APPEND config.py "\"${DIR}\", ")
endforeach()
file(APPEND config.py "]
")

# libraries
file(APPEND config.py "libraries = [")
foreach(DIR ${PYOPENCV_LIBRARIES})
    file(APPEND config.py "\"${DIR}\", ")
endforeach()
file(APPEND config.py "]
    
def revise_lib(lib):
    path, lib = op.split(op.normpath(lib))
    if not path=='':
        x = lib.rfind('.')
        if x >= 0: lib = lib[:x]
        if lib[:3]=='lib' : lib = lib[3:]
    return lib
    
libraries = [revise_lib(lib) for lib in libraries]

")


# ---------------------------------------------
# python extensions
# ---------------------------------------------

# PyOpenCV's "extras" library
file(GLOB_RECURSE EXTRA_SRCS package/extras/*.cpp)
add_library(pyopencv_extras SHARED ${EXTRA_SRCS})
set_target_properties(pyopencv_extras PROPERTIES 
    VERSION ${PYOPENCV_VERSION}
    SOVERSION ${PYOPENCV_VERSION}
    INSTALL_NAME_DIR ${PYOPENCV_INSTALL_PATH})
# set_target_properties(pyopencv_extras PROPERTIES 
    # ARCHIVE_OUTPUT_DIRECTORY ${PYOPENCV_INSTALL_PATH}
    # RUNTIME_OUTPUT_DIRECTORY ${PYOPENCV_INSTALL_PATH})
target_link_libraries(pyopencv_extras ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
if(WIN32)
    install(TARGETS pyopencv_extras RUNTIME DESTINATION ${PYOPENCV_INSTALL_PATH})
else()
    install(TARGETS pyopencv_extras DESTINATION ${PYOPENCV_INSTALL_PATH})
endif()

# write to config.py
file(APPEND config.py "pyopencv_extras_srcs = glob(op.join('package', 'extras', 'core', '*.cpp'))+glob(op.join('package', 'extras', 'sdopencv', '*.cpp'))

pyopencv_extras = Library('pyopencv.pyopencv_extras', 
    sources=pyopencv_extras_srcs, include_dirs=include_dirs,
    libraries=libraries, library_dirs=library_dirs,
    extra_compile_args=CXX_FLAGS, 
")
if(APPLE)
    file(APPEND config.py "extra_link_args=[ \"-install_name\", \"${PYOPENCV_INSTALL_PATH}/libpyopencv_extras.dylib\" ] + LINK_FLAGS)
")
else()
    file(APPEND config.py "extra_link_args=LINK_FLAGS)
")
endif()


# 'add_python_module' macro
macro(ADD_PYTHON_MODULE _NAME)
    file(GLOB ${_NAME}_SRCS package/${_NAME}_ext/*.cpp)
    add_library(${_NAME}_ext MODULE ${${_NAME}_SRCS})
    target_link_libraries(${_NAME}_ext pyopencv_extras ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
    # rpath
    set_target_properties(${_NAME}_ext PROPERTIES
	    INSTALL_RPATH ${PYOPENCV_INSTALL_PATH}
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE)
    set_target_properties(${_NAME}_ext PROPERTIES 
        PREFIX "" # There is no prefix even on UNIXes
        SUFFIX ${_modsuffix}) # The extension got from Python libraries
    install(TARGETS ${_NAME}_ext DESTINATION ${PYOPENCV_INSTALL_PATH})
    install(FILES package/pyopencv/${_NAME}.py DESTINATION ${PYOPENCV_INSTALL_PATH})
    set(PYTHON_MODULE_LIST ${PYTHON_MODULE_LIST} ${_NAME})
    
    # write to config.py
    file(APPEND config.py "${_NAME}_ext = Extension('pyopencv.${_NAME}_ext',
    sources=glob(op.join('package', '${_NAME}_ext', '*.cpp')), 
    include_dirs=include_dirs+[op.join('package', '${_NAME}_ext')],
    library_dirs=library_dirs, libraries=libraries+['pyopencv_extras'],
    extra_compile_args=CXX_FLAGS, extra_link_args=LINK_FLAGS)

")
endmacro(ADD_PYTHON_MODULE)

# CxType
set(WRAP_CXTYPE ON CACHE BOOL "Wrap the CxType component of OpenCV -- mandatory" FORCE)
if(WRAP_CXTYPE)
    add_python_module(cxtypes_h)
endif(WRAP_CXTYPE)

# testing
# get_target_property(TRIFEO2 cxtypes_h_ext LINK_FLAGS)
# message("hehe " ${TRIFEO2})

# CxCORE
set(WRAP_CXCORE ON CACHE BOOL "Wrap the CxCORE component of OpenCV")
if(WRAP_CXCORE)
    add_python_module(cxcore_h)
    add_python_module(cxcore_hpp_vec)
    add_python_module(cxcore_hpp_point)
    add_python_module(cxcore_hpp)
endif(WRAP_CXCORE)

# CxCORE
# set(WRAP_CXCORE ON CACHE BOOL "Wrap the CxCORE component of OpenCV -- mandatory" FORCE)
# if(WRAP_CXCORE)
    # add_python_module(cxtypes_h)
    # add_python_module(cxcore_h)
    # add_python_module(cxcore_hpp_vec)
    # add_python_module(cxcore_hpp_point)
    # add_python_module(cxcore_hpp)
# endif(WRAP_CXCORE)

# CV
set(WRAP_CV ON CACHE BOOL "Wrap the CV component of OpenCV")
if(WRAP_CV)
    add_python_module(cv_h)
    add_python_module(cv_hpp)
endif(WRAP_CV)

# CVAUX
set(WRAP_CVAUX ON CACHE BOOL "Wrap the CVAUX component of OpenCV")
if(WRAP_CVAUX)
    add_python_module(cvaux)
endif(WRAP_CVAUX)

# HighGUI
set(WRAP_HIGHGUI ON CACHE BOOL "Wrap the HighGUI component of OpenCV")
if(WRAP_HIGHGUI)
    add_python_module(highgui)
endif(WRAP_HIGHGUI)

# ML
set(WRAP_ML ON CACHE BOOL "Wrap the ML component of OpenCV")
if(WRAP_ML)
    add_python_module(ml)
endif(WRAP_ML)

# sdopencv
set(WRAP_SDOPENCV ON CACHE BOOL "Include extra code developed by Minh-Tri Pham, including classes like DifferentialImage, IntegralImage, and IntegralHistogram")
if(WRAP_SDOPENCV)
    add_python_module(sdopencv)
endif(WRAP_SDOPENCV)

# modules
foreach(MOD_NAME ${PYTHON_MODULE_LIST})
    file(APPEND package/pyopencv/config.py "from ${MOD_NAME} import *
")
endforeach()

# modules
file(APPEND config.py "extension_list = [pyopencv_extras, ")
foreach(MOD_NAME ${PYTHON_MODULE_LIST})
    file(APPEND config.py "${MOD_NAME}_ext, ")
endforeach()
file(APPEND config.py "]
")


# ---------------------------------------------
# installation
# ---------------------------------------------


install(FILES package/pyopencv/__init__.py package/pyopencv/config.py package/pyopencv/common.py package/pyopencv/info.py package/pyopencv/cxerror_h.py package/pyopencv/cvver_h.py package/pyopencv/interfaces.py DESTINATION ${PYOPENCV_INSTALL_PATH})
