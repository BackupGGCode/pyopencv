# Prerequisites #

## Supporting platforms ##

PyOpenCV is designed to be cross-platform. The official supporting platforms are Windows, Linux, and Mac OS. However, on Mac OS, I can mostly speculate since I barely have access to Mac OS at all.

## Software requirements ##

  1. [Python](http://www.python.org) version 2.5 or later. Python 3 is theoretically supported but not yet tested with.
  1. [NumPy](http://numpy.scipy.org) version 1.2.0 or later. You can install NumPy after PyOpenCV is installed. PyOpenCV requires and detects NumPy at run-time, not at install-time. There are many ways to install NumPy. For example, you could follow [SciPy's Installation page](http://www.scipy.org/Installing_SciPy/). It is not necessary to install SciPy though.
  1. [OpenCV](http://opencv.willowgarage.com/) version 2.x.

# Installation #

Like other packages, PyOpenCV can be installed from a binary distribution or from a source distribution.

## Installing from a binary distribution ##

This is the preferred way. Check the [Downloads](http://code.google.com/p/pyopencv/downloads/list) page to see if there is a binary distribution of PyOpenCV suitable for your platform.

## Installing from a source distribution ##

Right now, only gcc 4.x compilers are supported. The reason is that I have not tried other C++ compilers. If you have successfully built PyOpenCV using another C++ compiler, your knowledge contribution is much appreciated.

In short, there are three major steps:

  1. [Install OpenCV 2.x](Installation_OpenCV.md)
  1. [Install Boost.Python and bjam](Installation_Boost_Python.md) -- starting from version 2.1.0.wr.1.1.1, bjam is no longer required
  1. [Download, configure, build, and install PyOpenCV](Installation_PyOpenCV.md)

See also the [FAQ](FAQ.md) entries to avoid some installation issues.