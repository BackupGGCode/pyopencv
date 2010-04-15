#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

import re as _re

# -----------------------------------------------------------------------------------------------
# Some useful common-ground sub-routines
# -----------------------------------------------------------------------------------------------

def init_transformers(func_list):
    for fun in func_list:
        fun._transformer_creators = []
        fun._transformer_kwds = {}
        fun._args_docs = {}
        
def add_func_arg_doc(fun, arg, s=""):
    try:
        arg_doc = fun._args_docs[arg.name]
    except KeyError:
        arg_doc = []
        fun._args_docs[arg.name] = arg_doc
    arg_doc.append(s)
    
def wrap_str(s):
    ws_len = _re.match('\s*', s).end()
    ws = s[:ws_len]
    s = s[ws_len:]
    s2 = []
    while len(s) > 70:
        n = s[:70].rfind(' ')+1
        if n == 0:
            n = 70
        s2.append(ws+s[:n])
        s = s[n:]
    s2.append(ws+s)
    return s2
    
        
def _add_decl_boost_doc(decl, s="", append=True):
    if decl.documentation is not None:
        if append:
            decl.documentation += '\\\n    "\\n%s"' % s
        else:
            decl.documentation = ('"\\n%s"\n    ' % s) + decl.documentation
    else:
        decl.documentation ='"\\n%s"' % s

def add_decl_boost_doc(decl, s="", append=True, word_wrap=True):
    if word_wrap:
        s2 = wrap_str(s)
        if not append:
            s2.reverse()
        for s in s2:
            _add_decl_boost_doc(decl, s=s, append=append)
    else:
        _add_decl_boost_doc(decl, s=s, append=append)

dict_decl_name_to_desc = {
    # C -- cxcore -- Basic Structures
    ("::", "CvPoint"): ("2D point with integer coordinates (usually zero-based).", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Point2i instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvpoint"),
    ("::", "CvPoint2D32f"): ("2D point with floating-point coordinates.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Point2f instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvpoint2d32f"),
    ("::", "CvPoint3D32f"): ("3D point with floating-point coordinates.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Point3f instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvpoint3d32f"),
    ("::", "CvPoint2D64f"): ("2D point with double precision floating-point coordinates.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Point2d instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvpoint2d64f"),
    ("::", "CvPoint3D64f"): ("3D point with double precision floating-point coordinates.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Point3d instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvpoint3d64f"),
    ("::", "CvSize"): ("Pixel-accurate size of a rectangle.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Size2i instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvsize"),
    ("::", "CvSize2D32f"): ("Sub-pixel-accurate size of a rectangle.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Size2f instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvsize2d32f"),
    ("::", "CvScalar"): ("A container for 1-,2-,3- or 4-tuples of doubles.CvScalar is always represented as a 4-tuple..", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Scalar instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvscalar"),
    ("::", "CvTermCriteria"): ("Termination criteria for iterative algorithms.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class TermCriteria instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvtermcriteria"),
    ("::", "CvMat"): ("A multi-channel matrix.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Mat instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvmat"),
    ("::", "CvMatND"): ("Multi-dimensional dense multi-channel array.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Mat instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvmatnd"),
    ("::", "CvSparseMat"): ("Multi-dimensional sparse multi-channel array.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class SparseMat instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvsparsemat"),
    ("::", "IplImage"): ("IPL image header.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Mat instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#iplimage"),
    ("::", "CvArr"): ("Arbitrary array.", "Warning: This structure is obsolete. It exists only to support backward compatibility. Please use class Mat instead.", "http://opencv.willowgarage.com/documentation/basic_structures.html#cvarr"),
    
    # C -- cxcore -- Operations on Arrays
    ("::", "cvAbsDiff"): ("Calculates absolute difference between two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#absdiff"),
    ("::", "cvAbsDiffS"): ("Calculates absolute difference between an array and a scalar.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#absdiffs"),
    ("::", "cvAdd"): ("Computes the per-element sum of two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#add"),
    ("::", "cvAddS"): ("Computes the sum of an array and a scalar.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#adds"),
    ("::", "cvAddWeighted"): ("Computes the weighted sum of two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#addweighted"),
    ("::", "cvAnd"): ("Calculates per-element bit-wise conjunction of two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#and"),
    ("::", "cvAndS"): ("Calculates per-element bit-wise conjunction of an array and a scalar.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#ands"),
    ("::", "cvAvg"): ("Calculates average (mean) of array elements.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#avg"),
    ("::", "cvAvgSdv"): ("Calculates average (mean) of array elements.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#avgsdv"),
    ("::", "cvCalcCovarMatrix"): ("Calculates covariance matrix of a set of vectors.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#calccovarmatrix"),
    ("::", "cvCartToPolar"): ("Calculates the magnitude and/or angle of 2d vectors.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#carttopolar"),
    ("::", "cvCbrt"): ("Calculates the cubic root.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#cbrt"),
    ("::", "cvCmp"): ("Performs per-element comparison of two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#cmp"),
    ("::", "cvCmpS"): ("Performs per-element comparison of an array and a scalar.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#cmps"),
    ("::", "cvConvertScale"): ("Converts one array to another with optional linear transformation.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#convertscale"),
    ("::", "cvConvertScaleAbs"): ("Converts input array elements to another 8-bit unsigned integer with optional linear transformation.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#convertscaleabs"),
    ("::", "cvCvtScaleAbs"): ("Converts input array elements to another 8-bit unsigned integer with optional linear transformation.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#cvtscaleabs"),
    ("::", "cvCopy"): ("Copies one array to another.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#copy"),
    ("::", "cvCmp"): ("Performs per-element comparison of two arrays.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#cmp"),
    ("::", "cvCountNonZero"): ("Counts non-zero array elements.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#countnonzero"),
    ("::", "cvCrossProduct"): ("Calculates the cross product of two 3D vectors.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#crossproduct"),
    ("::", "cvDCT"): ("Performs a forward or inverse Discrete Cosine transform of a 1D or 2D floating-point array.", "http://opencv.willowgarage.com/documentation/operations_on_arrays.html#dct"),
    # TODO: append this

    # C -- cxcore -- Dynamic Structures
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cxcore -- Drawing Functions
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cxcore -- XML/YAML Persistence
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cxcore -- Clustering and Search in Multi-Dimensional Spaces
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cxcore -- Utility and System Functions and Macros
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cv -- Image Filtering
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cv -- Geometric Image Transformations
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C -- cv -- Miscellaneous Image Transformations
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # TODO: append the rest of the C reference

    # C -- highgui
    ("::", "cvConvertImage"): ("Converts one image to another with an optional vertical flip.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#convertimage"),
    ("::", "cvDestroyAllWindows"): ("Destroys all of the HighGUI windows.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#destroyallwindows"),
    ("::", "cvDestroyWindow"): ("Destroys a window.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#destroywindow"),
    ("::", "cvGetWindowHandle"): ("Gets the window's handle by its name.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#getwindowhandle"),
    ("::", "cvGetWindowName"): ("Gets the window's name by its handle.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#getwindowname"),
    ("::", "cvInitSystem"): ("Initializes HighGUI.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#initsystem"),
    ("::", "cvMoveWindow"): ("Sets the position of the window.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#movewindow"),
    ("::", "cvResizeWindow"): ("Sets the window size.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#resizewindow"),
    ("::", "cvSetMouseCallback"): ("Assigns callback for mouse events.", "http://opencv.willowgarage.com/documentation/c/user_interface.html#convertimage#setmousecallback"),

    # TODO: append the rest of the C reference

    
    # C++ -- cxcore -- Basic Structures
    ("::", "XXXX"): ("", ""),
    # TODO: append this
    
    # C++ -- cxcore -- Operations on Arrays
    ("cv", "abs"): ("Computes absolute value of each matrix element.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#abs"),
    ("cv", "absdiff"): ("Computes per-element absolute difference between 2 arrays or between array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#absdiff"),
    ("cv", "add"): ("Computes the per-element sum of two arrays or an array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#add"),
    ("cv", "addWeighted"): ("Computes the weighted sum of two arrays.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#addweighted"),
    ("cv", "bitwise_and"): ("Calculates per-element bit-wise conjunction of two arrays and an array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-bitwise-and"),
    ("cv", "bitwise_not"): ("Inverts every bit of array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-bitwise-not"),
    ("cv", "bitwise_or"): ("Calculates per-element bit-wise disjunction of two arrays and an array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-bitwise-or"),
    ("cv", "bitwise_xor"): ("Calculates per-element bit-wise 'exclusive or' operation on two arrays and an array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-bitwise-xor"),
    ("cv", "calcCovarMatrix"): ("Calculates covariation matrix of a set of vectors.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#calccovarmatrix"),
    ("cv", "cartToPolar"): ("Calculates the magnitude and angle of 2d vectors.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#carttopolar"),
    ("cv", "checkRange"): ("Checks every element of an input array for invalid values.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#checkrange"),
    ("cv", "compare"): ("Performs per-element comparison of two arrays or an array and scalar value.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#compare"),
    ("cv", "completeSymm"): ("Copies the lower or the upper half of a square matrix to another half.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#completesymm"),
    ("cv", "convertScaleAbs"): ("Scales, computes absolute values and converts the result to 8-bit.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#convertscaleabs"),
    ("cv", "countNonZero"): ("Counts non-zero array elements.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#countnonzero"),
    ("cv", "cubeRoot"): ("Computes cube root of the argument.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-cuberoot"),
    ("cv", "dct"): ("Performs a forward or inverse discrete cosine transform of 1D or 2D array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-dct"),
    ("cv", "dft"): ("Performs a forward or inverse Discrete Fourier transform of 1D or 2D floating-point array..", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-dft"),
    ("cv", "divide"): ("Performs per-element division of two arrays or a scalar by an array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-divide"),
    ("cv", "determinant"): ("Returns determinant of a square floating-point matrix.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-determinant"),
    ("cv", "eigen"): ("Computes eigenvalues and eigenvectors of a symmetric matrix.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-eigen"),
    ("cv", "exp"): ("Calculates the exponent of every array element.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-exp"),
    ("cv", "extractImageCOI"): ("Extract the selected image channel.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-extractimagecoi"),
    ("cv", "fastAtan2"): ("Calculates the angle of a 2D vector in degrees.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-fastatan2"),
    ("cv", "flip"): ("Flips a 2D array around vertical, horizontal or both axes.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-flip"),
    ("cv", "gemm"): ("Performs generalized matrix multiplication.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-gemm"),
    ("cv", "getConvertElem"): ("Returns conversion function for a single pixel.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-getconvertelem"),
    ("cv", "getOptimalDFTSize"): ("Returns optimal DFT size for a given vector size.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-getoptimaldftsize"),
    ("cv", "idct"): ("Computes inverse Discrete Cosine Transform of a 1D or 2D array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-idct"),
    ("cv", "idft"): ("Computes inverse Discrete Fourier Transform of a 1D or 2D array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-idft"),
    ("cv", "inRange"): ("Checks if array elements lie between the elements of two other arrays.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-inrange"),
    ("cv", "invert"): ("Finds the inverse or pseudo-inverse of a matrix.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-invert"),
    ("cv", "log"): ("Calculates the natural logarithm of every array element.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-log"),
    ("cv", "LUT"): ("Performs a look-up table transform of an array.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-lut"),
    ("cv", "magnitude"): ("Calculates magnitude of 2D vectors.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-magnitude"),
    ("cv", "Mahalanobis"): ("Calculates the Mahalanobis distance between two vectors.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-mahalanobis"),
    ("cv", "max"): ("Calculates per-element maximum of two arrays or array and a scalar.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-max"),
    ("cv", "mean"): ("Calculates average (mean) of array elements.", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-mean"),
    ("cv", "XXXXX"): (".", "http://opencv.willowgarage.com/documentation/cpp/operations_on_arrays.html#cv-xxxx"),
    # TODO: append this

    # TODO: append other C++ sections
    
    # C++ -- highgui
    ("cv", "createTrackbar"): ("Creates a trackbar and attaches it to the specified window.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-createtrackbar"),
    ("cv", "getTrackbarPos"): ("Returns the trackbar position.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-gettrackbarpos"),
    ("cv", "imshow"): ("Displays the image in the specified window.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-imshow"),
    ("cv", "namedWindow"): ("Creates a window.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-namedwindow"),
    ("cv", "setTrackbarPos"): ("Sets the trackbar position.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-settrackbarpos"),
    ("cv", "waitKey"): ("Waits for a pressed key.", "http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-waitkey"),
    ("cv", "imdecode"): ("Reads an image from a buffer in memory.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#cv-imdecode"),
    ("cv", "imencode"): ("Encode an image into a memory buffer.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#cv-imencode"),
    ("cv", "imread"): ("Loads an image from a file.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#cv-imread"),
    ("cv", "imwrite"): ("Saves an image to a specified file.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#cv-imwrite"),
    ("cv", "VideoCapture"): ("Class for video capturing from video files or cameras.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#VideoCapture"),
    ("cv", "VideoWriter"): ("Video writer class.", "http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#videowriter"),

    # TODO: append other C++ sections
}

def add_decl_desc(decl):
    try:
        # assume there are some docs for the declaration
        desc_list = dict_decl_name_to_desc[(decl.parent.name, decl.name)]
        desc_count = len(desc_list)-1
        reference = desc_list[desc_count]
    except KeyError:
        desc_list = None
        desc_count = 0
        reference = None
        
    try:
        # assume decl is a function
        for arg in decl._args_docs:
            add_decl_boost_doc(decl, "Argument '%s':" % arg)
            for z in decl._args_docs[arg]:
                add_decl_boost_doc(decl, "    "+z)            
    except AttributeError:
        pass
        
    if reference is not None:
        add_decl_boost_doc(decl, "    "+reference, False, word_wrap=False)
        add_decl_boost_doc(decl, "Reference:", False)    
    try:
        # assume decl is a function
        alias = decl.transformations[0].alias if len(decl.transformations) > 0 else decl.alias
        if alias != decl.name:
            add_decl_boost_doc(decl, "    "+decl.name, False)
            add_decl_boost_doc(decl, "Wrapped function:", False)
    except AttributeError:
        pass
        
    for i in xrange(desc_count-1, -1, -1):
        add_decl_boost_doc(decl, desc_list[i], False)
    
