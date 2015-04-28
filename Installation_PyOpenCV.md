# Downloading PyOpenCV #

Download a PyOpenCV source archive from the [Downloads](http://code.google.com/p/pyopencv/downloads/list) page, and extract it. Alternatively, use this command to anonymously check out the latest project source code:
```
svn checkout http://pyopencv.googlecode.com/svn/trunk/src/ pyopencv
```

If you want to check out PyOpenCV for OpenCV 2.0, however, check out at:
```
svn checkout http://pyopencv.googlecode.com/svn/branches/2.0.0/src/ pyopencv
```

# Configuring PyOpenCV #

## If you are installing PyOpenCV version 2.1.0.wr1.1.1 or later ##

PyOpenCV now supports building from cmake and setuptools.

### Configuring via cmake ###

First, you need have [cmake](http://www.cmake.org/) installed on your system. If you build OpenCV from source then cmake is probably already installed. If not, download and install cmake from http://www.cmake.org. On Ubuntu, you may want to issue the following command:

```
sudo apt-get install cmake cmake-gui
```

To configure using cmake, simply follow the standard process in cmake:
  * Create a custom build folder: `mkdir mybuild`
  * Change the current directory: `cd mybuild`
  * Run cmake or cmake-gui: `cmake ..` or `cmake-gui ..`
cmake has a capacity to detect OpenCV and Boost.Python on your system. If you receive an error message when running cmake, something must have been not configured correctly. Don't hesitate to raise your concern to the discussion group of the project.

If you are on Windows, you might want to use the MinGW generator instead, i.e.:

```
cmake -G "MinGW Makefiles" ..
```

### Configuring via setuptools ###

To configure using setuptools, simply invoke:
```
python setup.py config
```
This would invoke cmake automatically to configure. Internally, this is what the setup script does:
```
mkdir build
cd build
cmake ..
cd ..
```

A `config.py` file is then generated from cmake for PyOpenCV.

## If you are installing PyOpenCV version 2.1.0.wr1.1.0 or earlier ##

### Editing file `config.py` ###

PyOpenCV needs to be told where to find OpenCV 2.x and Boost. It needs to know the libraries to link against and the directories to search for those libraries. Put this text into file `config.py` located at PyOpenCV's source directory (potentially with adapted paths and libraries):
```
# OpenCV 2.x library
opencv_dir = <opencv_dir> # the root dir of your OpenCV build
opencv_include_dirs = [opencv_dir+"/include/opencv"]
opencv_library_dirs = [opencv_dir+"/lib"]
opencv_libraries = ["cvaux", "ml", "highgui", "cv", "cxcore"]
opencv_runtime_library_dirs = [opencv_dir+"/bin"]
opencv_runtime_libraries_to_be_bundled = []

# Boost library
boost_dir = <boost_dir> # the root dir of your Boost source tree
boost_include_dirs = [boost_dir]
boost_library_dirs = []
boost_libraries = []
boost_runtime_library_dirs = []
boost_runtime_libraries_to_be_bundled = []
```

This is the configuration file that provides PyOpenCV information about OpenCV and Boost libraries installed on your platform. The file is a Python script so the user can freely program to generate the required information automatically or manually.

Eventually, the file should have the following variables (each of which is a list/tuple of strings) exposed:

  * `opencv_include_dirs` == list of folders that contain OpenCV's include header files
  * `opencv_library_dirs` == list of folders that contain OpenCV's library files to be linked against (e.g. a folder containing files like cv210.lib, libcv210.a, libcv210.so, or libcv210.dylib)
  * `opencv_libraries` == list of library files that are to be linked against.
  * **Windows only:** `opencv_runtime_library_dirs` == list of folders that contain OpenCV's shared library files that are actually loaded at run-time (e.g. cv210.dll)
  * **Windows only:** `opencv_runtime_libraries_to_be_bundled` == list of shared library files that are actually loaded at run-time. If this variable is an empty list (i.e. `[]`), all the folders specified in the 'opencv\_runtime\_library\_dirs' variable are inserted at the front of the PATH environment whenever PyOpenCV is imported. Otherwise, these shared library files are bundled with PyOpenCV at install-time.

  * `boost_include_dirs` == list of folders that contain Boost's include header files. The first item of the list must be the root path of Boost.
  * `boost_library_dirs` == list of folders that contain Boost.Python's library files to be linked against (e.g. a folder containing files like libboostpython.a or boost\_python-mgw44-mt.lib). This variable is ignored if bjam is used as the compiler.
  * `boost_libraries` == list of library files that are to be linked against. This variable is ignored if bjam is used as the compiler.
  * **Windows only:** `boost_runtime_library_dirs` == list of folders that contain Boost.Python's shared library files that are actually loaded at run-time (e.g. boost\_python-mgw44-mt-1\_40.dll). This variable is ignored if bjam is used as the compiler.
  * **Windows only:** `boost_runtime_libraries_to_be_bundled` == list of shared library files that are actually loaded at run-time. If this variable is an empty list (i.e. `[]`), all the folders specified in the 'boost\_runtime\_library\_dirs' variable are inserted at the front of the PATH environment whenever PyOpenCV is imported. Otherwise, these shared library files are bundled with PyOpenCV at install-time. This variable is ignored if bjam is used as the compiler.

Here are some examples of `config.py`:

  * **On Windows:**

```
# OpenCV 2.1.0 library, built with MinGW
opencv_dir = "C:/MinGW/OpenCV"
opencv_include_dirs = [opencv_dir+"/include/opencv"]
opencv_library_dirs = [opencv_dir+"/lib"]
opencv_libraries = ["cvaux210.dll", "ml210.dll", "highgui210.dll", "cv210.dll", "cxcore210.dll"]
opencv_runtime_library_dirs = [opencv_dir+"/bin"]
opencv_runtime_libraries_to_be_bundled = []

# Boost library
boost_dir = "C:/boost_1_41_0"
boost_include_dirs = [boost_dir]
boost_library_dirs = []
boost_libraries = []
boost_runtime_library_dirs = []
boost_runtime_libraries_to_be_bundled = []
```

  * **On Linux:**

```
# OpenCV 2.1.0 library
opencv_dir = "/usr/local"
opencv_include_dirs = [opencv_dir+"/include/opencv"]
opencv_library_dirs = [opencv_dir+"/lib"]
opencv_libraries = ["highgui", "ml", "cvaux", "cv", "cxcore"]
opencv_runtime_library_dirs = []
opencv_runtime_libraries_to_be_bundled = []

# Boost library
boost_dir = "/home/user/boost_1_41_0"
boost_include_dirs = [boost_dir]
boost_library_dirs = []
boost_libraries = []
boost_runtime_library_dirs = []
boost_runtime_libraries_to_be_bundled = []
```

# Building and Installing PyOpenCV #

## If you are installing PyOpenCV version 2.1.0.wr1.1.1 or later ##

### Building and installing via cmake ###

To build and PyOpenCV via cmake, go to your custom build folder and issue the command
```
make
make install
```

**On Windows**: if you are using MinGW as the compiler for PyOpenCV, you may want to use `mingw32-make` instead:
```
mingw32-make
mingw32-make install
```

**On Linux**: make sure that you have the privilege to install on your system. For instance, on Ubuntu:
```
make
sudo make install
```

### Building and installing via setuptools ###

To build, issue the command:
```
python setup.py build
```

**On Windows**: you may want to specify MinGW as the compiler, i.e.:
```
python setup.py build -cmingw32
```

Depending the number of modules you would like to install, it may take up to 45 minutes to build. Take a rest and have some coffee or green tea.

To install, issue the command:
```
python setup.py install
```

**On Linux**: make sure you have the privilege to install on your system. For instance, on Ubuntu:
```
sudo python setup.py install
```


## If you are installing PyOpenCV version 2.1.0.wr1.1.0 or earlier ##

Now that you have obtained all the necessary libraries, the remaining steps are straight forward.

To build PyOpenCV, issue the command
```
python setup.py build
```
It should invoke bjam (PyOpenCV's default choice) to build a Python extension for PyOpenCV. The whole process of building PyOpenCV should take some time between 10 minutes and 45 minutes. Go grab yourself a cup of coffee, sit back, and take a break.

To install PyOpenCV, issue the command
```
python setup.py install
```