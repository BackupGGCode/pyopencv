PyOpenCV relies on [Boost.Python](http://www.boost.org/doc/libs/release/libs/python/doc/) (and partially on Boost.MPL) to wrap C/C++ code. In version 2.1.0.wr1.1.0 and earlier, PyOpenCV relies on bjam to build Python extensions that use Boost.Python. However, starting from version 2.1.0.wr1.1.1, bjam is no longer required by PyOpenCV (but you may still need it to install Boost.Python). Instead, PyOpenCV relies on CMake to detect both OpenCV and Boost.Python installed on your system. In these versions, you can use either setuptools or cmake to build PyOpenCV.

# Installing Boost.Python if you are installing PyOpenCV version 2.1.0.wr1.1.1 or later #

Simply follow the [standard instructions](http://www.boost.org/doc/libs/release/libs/python/doc/building.html#basic-procedure) to install Boost.Python on your system. However, there are a few important notices:
  1. You must make sure that a shared version of Boost.Python is installed (e.g. by specifying option `link=shared` when invoking `bjam`). Python extensions linked against a shared version of Boost.Python can share their exposed declarations with each other. PyOpenCV is now modularised into a few extensions, allowing the compilation to reduce significantly and enabling the user an option to select which modules to install. If you use a static version of Boost.Python, the declarations are not shared and PyOpenCV will fail.
  1. The version of Boost.Python must be at last 1.41.0 or 1.40.0 with patches. There is a bug in Boost.Python under Python 2.6.3 and later that has only been fixed very recently in Boost 1.41.0 (thanks to Guy K. Kloss for having pointed it out).
  1. Remember that the preferred C++ compiler is gcc and I have not tested with other C++ compiler. Therefore, if you build Boost.Python from source, you should specify the option `toolset=gcc` when invoking bjam.
  1. If, for some reason, PyOpenCV cannot detect your Boost.Python installation, make sure you have the `ROOT_BOOST` environment variable point to the root directory of the Boost source distribution.

## Examples ##

### Building Boost.Python from source ###
  1. Download Boost source distribution and install bjam. If you are not sure how to do so, see the instructions in the next section.
  1. Run the command:
```
    bjam toolset=gcc link=shared --with-python
```

### Installing Boost.Python from Linux distributions ###

Some Linux distributions come with a pre-compiled Boost.Python shared library. Just follow the standard procedure to install it. For example, on Ubuntu:
```
    sudo apt-get install libboost-python-dev
```


# Installing Boost and bjam if you are installing PyOpenCV version 2.1.0.wr1.1.0 or earlier #

## Getting Boost ##

Please follow the instructions below to get the Boost library:

  1. Go to http://www.boost.org.
  1. Download [Boost source distribution](http://sourceforge.net/projects/boost/files/boost/) version [1.41.0](http://sourceforge.net/projects/boost/files/boost/1.41.0/) or later. There is a bug in using Boost.Python under Python 2.6.3 and later that has only been fixed very recently in Boost 1.41.0 (thanks to Guy K. Kloss for having pointed it out).
  1. Extract the source distribution to somewhere on your platform. We refer to the extracted folder as `<boost_dir>`. It is not necessary to install Boost.
  1. **On Linux:** in order for Boost.Python to compile, you may need to install the `python-dev` package, if it has not been installed.

Note that, right now you should rely on a Boost source distribution to install rather on any packages of Boost that have been pre-compiled for your platform. Unless you understand how PyOpenCV is compiled and linked against Boost libraries (which is fairly straightforward), just don't push your luck. This technical issue is mainly due to that I don't have time to tweak the setup file to deal with those situations yet. But I will try to do so.

Also, once you have finished installing PyOpenCV, you can safely delete the Boost source distribution. Currently, PyOpenCV is only and statically linked against Boost.Python. Last time I checked, the binary size of PyOpenCV linked against a static Boost Python library and that against a shared Boost Python library were almost the same, only less than 1% different. Therefore, to avoid linking issues, I decided to make linking PyOpenCV against a static Boost.Python library as the default option. You can easily modify the setup file to link against a shared Boost.Python library though.

## Getting bjam ##

  1. PyOpenCV relies on Boost.Python to interface with OpenCV, and on Boost.Jam (or bjam) to compile its internal Python extension. The recommended way to get Boost.Jam is to [download a prebuilt executable](http://sourceforge.net/project/showfiles.php?group_id=7586&package_id=72941) from SourceForge. If a prebuilt executable is not provided for your platform or you are using Boost's sources in an unreleased state, it may be necessary to [build bjam from source](http://www.boost.org/doc/tools/build/doc/html/jam/building.html) included in the Boost source tree.
  1. To install Boost.Jam, copy the executable, called `bjam` or `bjam.exe` to a location accessible in your PATH. Go to the `<boost_dir>/tools/build/v2` directory and run `bjam --version`. You should see something like:
```
            Boost.Build V2 (Milestone N)
            Boost.Jam xx.xx.xx 
```
  1. **On Ubuntu**: alternatively, run `sudo apt-get install bjam` to install bjam.