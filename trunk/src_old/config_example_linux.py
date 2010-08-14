#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

# This is the configuration file that provides PyOpenCV information about OpenCV and Boost libraries installed on the user's platform.
# The file is a Python sript so the user can freely program to generate the required information automatically or manually.

# Eventually, the file should have the following variables (each of which is a list/tuple of strings) exposed:

#  - opencv_include_dirs == list of folders that contain OpenCV's include header files
#  - opencv_library_dirs == list of folders that contain OpenCV's library files to be linked against (e.g. a folder containing files like cv200.lib, libcv200.dll.a, or libcv200.a)
#  - opencv_libraries == list of library files that are to be linked against. 
#  - opencv_runtime_library_dirs == list of folders that contain OpenCV's shared library files that are actually loaded at run-time (e.g. cv200.dll, libcv200.so, or libcv200.dylib)
#  - opencv_runtime_libraries_to_be_bundled == list of shared library files that are actually loaded at run-time. If this variable is an empty list (i.e. []), all the folders specified in the 'opencv_runtime_library_dirs' variable are inserted at the front of the PATH environment whenever PyOpenCV is imported. Otherwise, these shared library files are bundled with PyOpenCV at install-time.

#  - boost_include_dirs == list of folders that contain Boost's include header files. The first item of the list must be the root path of Boost.
#  - boost_library_dirs == list of folders that contain Boost.Python's library files to be linked against (e.g. a folder containing files like libboostpython.a or boost_python-mgw44-mt.lib). This variable is ignored if bjam is used as the compiler.
#  - boost_libraries == list of library files that are to be linked against. This variable is ignored if bjam is used as the compiler.
#  - boost_runtime_library_dirs == list of folders that contain Boost.Python's shared library files that are actually loaded at run-time (e.g. boost_python-mgw44-mt-1_40.dll). This variable is ignored if bjam is used as the compiler.
#  - boost_runtime_libraries_to_be_bundled == list of shared library files that are actually loaded at run-time. If this variable is an empty list (i.e. []), all the folders specified in the 'boost_runtime_library_dirs' variable are inserted at the front of the PATH environment whenever PyOpenCV is imported. Otherwise, these shared library files are bundled with PyOpenCV at install-time. This variable is ignored if bjam is used as the compiler.

import os
from glob import glob

# OpenCV 2.x library, to be linked against using bjam+gcc
opencv_dir = "/usr/local"
opencv_include_dirs = [opencv_dir+"/include/opencv"]
opencv_library_dirs = [opencv_dir+"/lib"]
opencv_libraries = ["highgui", "ml", "cvaux", "cv", "cxcore"]
opencv_runtime_library_dirs = []
opencv_runtime_libraries_to_be_bundled = []

# Boost library's source distribution, to be linked against using bjam+gcc and bundled with
boost_include_dirs = ["/home/inteplus/boost_1_41_0"]
boost_library_dirs = []
boost_libraries = []
boost_runtime_library_dirs = []
boost_runtime_libraries_to_be_bundled = []
