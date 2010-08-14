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

# Default configuration on 32-bit Windows

# OpenCV 2.x library, to be linked against using gcc
opencv_include_dirs = ["C:/Program Files/OpenCV/include/opencv"]
opencv_library_dirs = ["C:/Program Files/OpenCV/lib"]
opencv_runtime_library_dirs = ["C:/Program Files/OpenCV/bin"]
opencv_libraries = ["cvaux200.dll", "ml200.dll", "highgui200.dll", "cv200.dll", "cxcore200.dll"]
opencv_runtime_libraries_to_be_bundled = []

# Boost library's source distribution, to be linked against using gcc
boost_include_dirs = ["C:/boost_1_40_0"]
boost_library_dirs = ["C:/boost_1_40_0/stage/lib"]
boost_runtime_library_dirs = ["C:/boost_1_40_0/stage/lib"]
boost_libraries = ["boost_python-mgw44"]
boost_runtime_libraries_to_be_bundled = []