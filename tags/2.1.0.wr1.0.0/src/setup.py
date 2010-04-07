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

from glob import glob
import sys
import os
import os.path as OP

try:
    import config as C
except ImportError:
    print "You must first create/edit file 'config.py' in this setup folder before you can run 'setup.py'. Just copy file 'config_example_win32.py' or 'config_example_linux.py' to file 'config.py', Edit it to match with your platform, and run 'setup.py' again."
    sys.exit(-1)


import ez_setup
ez_setup.use_setuptools()

import bjamcompiler as BJ
BJ.boost_dir = C.boost_include_dirs[0]

from setuptools import setup, find_packages, Extension

pyopencvext = Extension('pyopencv.pyopencvext',
    sources=glob(OP.join('pyopencv', 'pyopencvext', '*.cpp'))+\
        glob(OP.join('pyopencv', 'pyopencvext', 'sdopencv', '*.cpp'))+\
        glob(OP.join('pyopencv', 'pyopencvext', 'core', '*.cpp')),
    include_dirs=C.opencv_include_dirs+C.boost_include_dirs+['pyopencv', 
        OP.join('pyopencv', 'pyopencvext'), OP.join('pyopencv', 'pyopencvext', 'numpy_include'),
        OP.join('pyopencv', 'pyopencvext', 'core'), OP.join('pyopencv', 'pyopencvext', 'sdopencv')],
    library_dirs=C.opencv_library_dirs+C.boost_library_dirs,
    libraries=C.opencv_libraries+C.boost_libraries,
    runtime_library_dirs=C.opencv_runtime_library_dirs+C.boost_runtime_library_dirs,
)

# prepare pyopencv/config.py
f = open('pyopencv/config.py', 'wt')
f.write('''#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# This file was generated by setup.py. Please do not modify it unless you know what you are doing.

path_ext = []
''')
if not C.opencv_runtime_libraries_to_be_bundled:
    for path in C.opencv_runtime_library_dirs:
        f.write('path_ext.append("%s")\n' % BJ.mypath(OP.abspath(path)))
if not C.boost_runtime_libraries_to_be_bundled:
    for path in C.boost_runtime_library_dirs:
        f.write('path_ext.append("%s")\n' % BJ.mypath(OP.abspath(path)))
f.close()

import distutils.file_util as D

def find_libraries(library_dirs, libraries):
    files = []
    for lib in libraries:
        for dir in library_dirs:
            if OP.exists(OP.join(dir, lib)):
                D.copy_file(OP.join(dir, lib), 'pyopencv')
                files.append(lib)
                break
        else:
            raise IOError("Library %s not found." % lib)
    return files

bundled_files = find_libraries(C.opencv_runtime_library_dirs, C.opencv_runtime_libraries_to_be_bundled) + \
    find_libraries(C.boost_runtime_library_dirs, C.boost_runtime_libraries_to_be_bundled)

setup(
    name = "pyopencv",
	version = '2.1.0.wr1.0.0',
	description = DOCLINES[0],
	author = 'Minh-Tri Pham',
	author_email = 'pmtri80@gmail.com',
	url = 'http://code.google.com/p/pyopencv/',
	license = 'New BSD License',
	platforms = 'OS Independent, Windows, Linux, MacOS',
	classifiers = filter(None, CLASSIFIERS.split('\n')),
	long_description = "\n".join(DOCLINES[2:]),
    ext_modules=[pyopencvext],
    # install_requires = ['numpy>=1.2.0'],
    package_data = {'pyopencv': bundled_files, '': ['AUTHORS', 'ChangeLog', 'COPYING', 'README', 'THANKS', 'TODO']},
    # include_package_data = True,
    packages = find_packages(),
)

