# General questions #

### Which version of OpenCV is PyOpenCV compatible with? ###

PyOpenCV is compatible with OpenCV version 2.0 and 2.1.0. The main line of development currently supports OpenCV 2.1.0, which is available in the `trunk` folder of the svn repository. The line of development that supports OpenCV 2.0 is available in the `branches/2.0.0` folder of the svn repository.

### Is PyOpenCV 64-bit compatible? ###

PyOpenCV is now compilable and running on a 64-bit Ubuntu 10.4 platform and a Mac OS X 10.6 platform (thanks Attila). Other platforms have not been tested on.

### Is PyOpenCV compatible with Python 3? ###

Almost all of the source code of PyOpenCV is compatible with Python 3. Boost.Python and NumPy are now both compatible with Python 3. The only remaining show stopper is setuptools. There is an alternative package to setuptools called 'distribute' that is compatible with Python 3. I will try to see if we can replace setuptools with it and make PyOpenCV fully compatible with Python 3.

### Declaration X or declaration X of class Y is not exposed. Can I have it exposed? ###

If the declaration you are looking for belongs to the old C interface, check if there is an equivalent declaration in the new C++ interface. Otherwise, find the related issue in the "Issues" tab (or file one if there is none), and post your request. I will try to fix the issue when I have time. Certainly, the more requests there are, the higher priority the issue will receive.

### Is there any efficient way to find the relationship between an API in C and in PyOpenCV? For example, how can I find the correspondence of the method, cvNormalizeHist()? ###

If an OpenCV function in C is named as cvAbcXyz(), it is typically renamed as abcXyz() in C++. Just look for function abcXyz() in PyOpenCV. In some rare cases, a character or a short string is appended to the function's name to define a different transform of the function.

If you do not see abcXyz(), then either an equivalent function/class in C++ exists or the function is not yet exposed. In the latter case, please file an issue on the project's Issues tab. I will try to resolve it when I have time.

# Questions related to PyOpenCV version 2.1.0.wr1.1.0 and earlier #

### On Linux, when building PyOpenCV, I receive an error: "...error: duplicate initialization of gcc with the following parameters:..." ###

It appears that there is a file named `site-config.jam` located somewhere in your home folder. Try running `bjam` manually with `--debug-configuration` and look for a line that
looks like:

```
    notice: Loading site-config configuration file site-config.jam from
    path/to/site-config.jam . 
```

and uncomment a line that looks like

```
    using gcc ;
```

in that file. Then, build PyOpenCV by running `setup.py` again.