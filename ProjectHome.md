This package takes a completely new and different approach in wrapping [OpenCV](http://opencv.willowgarage.com) from traditional swig-based and ctypes-based approaches. It is intended to be a successor of [ctypes-opencv](http://code.google.com/p/ctypes-opencv/) and to provide Python bindings for OpenCV 2.x. Ctypes-based approaches like ctypes-opencv, while being very flexible at wrapping functions and structures, are weak at wrapping OpenCV's C++ interface. On the other hand, swig-based approaches flatten C++ classes and create countless memory management issues. In PyOpenCV, we use Boost.Python, a C++ library which enables seamless interoperability between C++ and Python. PyOpenCV offers a better solution than both ctypes-based and swig-based wrappers. Its main features include:
  * A Python interface almost the same as the new C++ interface of OpenCV 2.x, including features that are available in the existing C interface but not yet in the C++ interface.
  * Bindings for all major components of OpenCV 2.x: CxCORE (almost complete), CxFLANN (complete), Cv (complete), CvAux (C++ part almost complete, C part in progress), CvVidSurv (complete), HighGui (complete), and ML (complete).
  * Access to C++ data structures in Python.
  * Elimination of memory management issues. The user never has to worry about memory management.
  * Ability to convert between OpenCV's Mat and arrays used in wxWidgets, PyGTK, and PIL.

To the best of our knowledge, PyOpenCV is the largest wrapper among existing Python wrappers for OpenCV. It exposes to Python 200+ classes and 500+ free functions of OpenCV 2.x, including those instantiated from templates.

In addition, we use [NumPy](http://numpy.scipy.org) to provide fast indexing and slicing functionality for OpenCV's dense data types like Vec-like, Point-like, Rect-like, Size-like, Scalar, Mat, and MatND, and to offer the user an option to work with their multi-dimensional arrays in NumPy. It is well-known that NumPy is one of the best packages (if not the best) for dealing with multi-dimensional arrays in Python. OpenCV 2.x provides a new C++ generic programming approach for matrix manipulation (i.e. MatExpr). It is a good attempt in C++. However, in Python, a package like NumPy is without a doubt a better solution. By incorporating NumPy into PyOpenCV to replace OpenCV 2.x's MatExpr approach, we bring OpenCV and NumPy closer together, and offer a package that inherits the best of both worlds: fast computer vision functionality (OpenCV) and fast multi-dimensional array computation (NumPy).

## Installation ##

> General instructions for installing PyOpenCV can be found at the [Installation](Installation.md) page.

## Documentation ##

> Check out the [Documentation](Documentation.md) page.

## Examples ##

### Pedestrian Detection ###
```
from pyopencv import *

img = imread('people.jpg')
hog = HOGDescriptor()
hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector())
for r in hog.detectMultiScale(img, 0, Size(8,8), Size(24,16), 1.05, 2):
    r.x += round(r.width*0.1)
    r.y += round(r.height*0.1)
    r.width = round(r.width*0.8)
    r.height = round(r.height*0.8)
    rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 1)

namedWindow("people detector", 1)
imshow("people detector", img)
waitKey(0)
```

![http://mtpham.sharkdolphin.com/open_source/pyopencv/demo/pedestrian_detection.jpg](http://mtpham.sharkdolphin.com/open_source/pyopencv/demo/pedestrian_detection.jpg)

### K-Means Clustering ###
```
from pyopencv import *
import numpy.random as NR
MAX_CLUSTERS=5

if __name__ == "__main__":

    color_tab = [CV_RGB(255,0,0),CV_RGB(0,255,0),CV_RGB(100,100,255), CV_RGB(255,0,255),CV_RGB(255,255,0)]
    img = Mat(Size(500, 500), CV_8UC3)
    rng = RNG()
    namedWindow( "clusters", 1 )
        
    while True:
        cluster_count = rng.as_uint()%(MAX_CLUSTERS-1) + 2
        
        # generate random sample from multigaussian distribution
        points = NR.randn(cluster_count, rng.as_uint()%200 + 1, 2)*(img.cols, img.rows)*0.1
        for k in range(cluster_count):
            points[k] += (rng.as_uint()%img.cols, rng.as_uint()%img.rows)
        sample_count = points.size/2
        points = asMat(points.reshape(sample_count, 1, 2).astype('float32'))
        randShuffle( points )
        
        # K Means Clustering
        clusters = Mat(points.size(), CV_32SC1)
        compact, centers = kmeans(points, cluster_count, clusters, 
            TermCriteria(TermCriteria.EPS+TermCriteria.MAX_ITER, 10, 1.0), 3, KMEANS_RANDOM_CENTERS)

        img.setTo(0)
        pts = points[:].reshape(sample_count, 2).astype('int')
        for i in range(sample_count):
            circle(img, asPoint(pts[i]), 2, color_tab[clusters[i,0]], CV_FILLED, CV_AA, 0)
        
        imshow( "clusters", img )

        if '%c' % (waitKey(0) & 255) in ['\x1b','q','Q']: # 'ESC'
            break
```

![http://mtpham.sharkdolphin.com/open_source/pyopencv/demo/kmeans.png](http://mtpham.sharkdolphin.com/open_source/pyopencv/demo/kmeans.png)

Check out the demo package on the Downloads tab for other examples.

## Development ##

At the moment, PyOpenCV is at the beta stage. However, it is rather stable and only minor issues are remaining. I am the sole author/developer of the project. I use [Py++](http://www.language-binding.net/pyplusplus/pyplusplus.html) to generate the source code, but I can only work in my spare time. I constantly look for partners to develop the project. If you would like to join in, please let me know.

Currently, the development of PyOpenCV is focused on (in descending order of priority):

  1. Documentation: i.e. improving the docstrings and migrating from epydoc to sphinx.
  1. Demonstration: i.e. adding more demo code.
  1. Adding remaining features: Some pointer member variables and functions with pointer arguments are harder to expose than others. Most of them have been exposed, but some of them are still missing.
  1. Compatibility with Python 3: PyOpenCV itself is almost Python 3 ready. NumPy has been Python 3 ready recently. All we have to do is to wait for setuptools to be Python 3 ready. An alternative is to replace setuptools with distribute. But I have not got time to try it out.

## Bugs and Commentary ##

> Please send information on issues of usage to Minh-Tri Pham <pmtri80@gmail.com>, post a message to [PyOpenCV and ctypes-opencv's discussion group](http://groups.google.com/group/ctypes-opencv), or file an issue on the Issues tab.

## Acknowledgment ##

> I would like to thank everyone in the discussion group of the project, as well as those who have contacted me privately via email, for their invaluable discussion and idea/code contributions.

> Thanks also go to the people at Intel and Willow Garage for having invented and developed OpenCV. I have been using it extensively in my research.

> Until an actual paper for PyOpenCV is available, if you use PyOpenCV in your research, please cite the following paper in your publication (although it is entirely voluntary):

```
@inproceedings{Pham2010,
  author = {Minh-Tri Pham and Yang Gao and Viet-Dung D. Hoang and Tat-Jen Cham},
  title = {Fast Polygonal Integration and Its Application in Extending Haar-like Features to Improve Object Detection},
  booktitle = {Proc. IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2010},
  address = {San Francisco, California},
  month = {Jun}
}
```
