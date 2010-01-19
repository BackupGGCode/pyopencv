#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.0 using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

# OpenCV 2.0 library
opencv_include_dirs = ["M:/programming/packages/opencv/build/2.0/include/opencv"]
opencv_library_dirs = ["M:/programming/packages/opencv/build/2.0_for_python/lib"]
opencv_runtime_library_dirs = ["M:/programming/packages/opencv/build/2.0_for_python/bin"]
opencv_libraries = ["cvaux200.dll", "ml200.dll", "highgui200.dll", "cv200.dll", "cxcore200.dll"]

# Boost.Python library
boost_include_dirs = ["M:/programming/packages/scipack/boost/boost_1_40_0"]
# boost_library_dirs = ["M:/programming/packages/scipack/boost/boost_1_40_0/bin.v2/libs/python/build/gcc-mingw-4.4.0/release/link-static"]
boost_library_dirs = ["M:/programming/packages/scipack/boost/boost_1_40_0/stage/lib"]
boost_runtime_library_dirs = ["M:/programming/packages/scipack/boost/boost_1_40_0/stage/lib"]
# boost_libraries = ["libboost_python-mgw44-1_40"]
boost_libraries = ["boost_python-mgw44"]
