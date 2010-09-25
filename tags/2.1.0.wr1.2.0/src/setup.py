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
"""PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

PyOpenCV brings Willow Garage's Open Source Computer Vision Library
(OpenCV) verion 2.x to Python. The package takes a completely new and
different approach in wrapping OpenCV from traditional swig-based and
ctypes-based approaches. It is intended to be a successor of
ctypes-opencv and to provide Python bindings for OpenCV 2.x.
Ctypes-based approaches like ctypes-opencv, while being very flexible at
wrapping functions and structures, are weak at wrapping OpenCV's C++
interface. On the other hand, swig-based approaches flatten C++ classes
and create countless memory management issues. In PyOpenCV, we use
Boost.Python, a C++ library which enables seamless interoperability
between C++ and Python. PyOpenCV will offer a better solution than both
ctypes-based and swig-based wrappers. Its main features include:

    * A Python interface similar to the new C++ interface of OpenCV 2.x,
      including features that are available in the existing C interface
      but not yet in the C++ interface.
    * Access to C++ data structures in Python.
    * Elimination of memory management issues. The user never has to
      worry about memory management.
    * Ability to convert between OpenCV's Mat and arrays used in
      wxWidgets, PyGTK, and PIL.
    * OpenCV extensions: classes DifferentialImage, IntegralImage, and
      IntegralHistogram.

To the best of our knowledge, PyOpenCV is the largest wrapper among
existing Python wrappers for OpenCV. It exposes to Python 200+ classes
and 500+ free functions of OpenCV 2.x, including those instantiated from
templates.

In addition, we use NumPy to provide fast indexing and slicing
functionality to OpenCV's dense data types like Vec-like, Point-like,
Rect-like, Size-like, Scalar, Mat, and MatND, and to offer the user an
option to work with their multi-dimensional arrays in NumPy. It is
well-known that NumPy is one of the best packages (if not the best) for
dealing with multi-dimensional arrays in Python. OpenCV 2.x provides a
new C++ generic programming approach for matrix manipulation (i.e.
MatExpr). It is a good attempt in C++. However, in Python, a package
like NumPy is without a doubt a better solution. By incorporating NumPy
into PyOpenCV to replace OpenCV 2.x's MatExpr approach, we seek to bring
OpenCV and NumPy closer together, and offer a package that inherits the
best of both world: fast computer vision functionality (OpenCV) and fast
multi-dimensional array computation (NumPy).

"""

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: OS Independent
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python
Topic :: Multimedia :: Graphics
Topic :: Multimedia :: Video
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Human Machine Interfaces
Topic :: Software Development :: Libraries :: Python Modules
"""

import distutils_hack

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages, Extension, Library

from glob import glob
import sys
import os
import os.path as op
import distutils.spawn as ds
import distutils.dir_util as dd

try:
    import config as C
except ImportError: # no config.py file found
    if ds.find_executable('cmake') is None:
        print "Error: unable to configure PyOpenCV!"
        print
        print "Starting from version 1.1.1 OpenCV 2.1.0, PyOpenCV relies on the"
        print "CMake build tool (http://www.cmake.org/) to configure. However,"
        print "CMake is not found in your system. Please install CMake before"
        print "running the setup file. "
        print 
        print "Once CMake is installed, you can also manually configure PyOpenCV"
        print "by running the following commands:"
        print "    mkdir build"
        print "    cd build"
        print "    cmake .."
        print "    cd .."
        sys.exit(-1)
        
    print "Configuring PyOpenCV via CMake..."
    cur_dir = os.getcwd()
    new_dir = op.join(op.split(__file__)[0], 'build')
    dd.mkpath(new_dir)
    os.chdir(new_dir)
    try:
        ds.spawn(['cmake', '..'])
    except ds.DistutilsExecError:
        print "Error: error occurred while running CMake to configure PyOpenCV."
        print "You may want to manually configure PyOpenCV by running cmake's tools:"
        print "    mkdir build"
        print "    cd build"
        print "    cmake-gui ..    OR    cmake .."
        print "    cd .."
        sys.exit(-1)
    os.chdir(cur_dir)
    import config as C

setup(
    name = "pyopencv",
	version = C.PYOPENCV_VERSION,
	description = DOCLINES[0],
	author = 'Minh-Tri Pham',
	author_email = 'pmtri80@gmail.com',
	url = 'http://code.google.com/p/pyopencv/',
	license = 'New BSD License',
	platforms = 'OS Independent, Windows, Linux, MacOS',
	classifiers = filter(None, CLASSIFIERS.split('\n')),
	long_description = "\n".join(DOCLINES[2:]),
    ext_modules=C.extension_list,
    install_requires = ['numpy>=1.2.0'],
    package_data = {'pyopencv': ['*.dll']},
    include_package_data = True,
    # zip_safe = (os.name!='nt'), # thanks to ffmpeg dependency
    package_dir={'':'package'},
    packages = find_packages('package'),
)

