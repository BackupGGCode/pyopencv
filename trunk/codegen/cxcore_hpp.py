#!/usr/bin/env python
# pyopencv - A Python wrapper for OpenCV 2.0 using Boost.Python and ctypes

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------


def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cxcore.hpp
#=============================================================================

Size2i = Size

    ''')
    
    #=============================================================================
    # Structures
    #=============================================================================
    
    # Vec et al
    mb.class_('::cv::Vec<int, 4>').rename('Vec4i')
    zz = mb.classes(lambda z: z.name.startswith('Vec<'))
    for z in zz:
        z.include()
        z.decl('val').exclude() # use operator[] instead
        z.operator(lambda x: x.name.endswith('::CvScalar')).rename('as_CvScalar')
        
    # Complex et al
    zz = mb.classes(lambda z: z.name.startswith('Complex<'))
    for z in zz:
        z.include()
        z.decls(lambda t: 'std::complex' in t.decl_string).exclude() # no std::complex please
    
    # Point et al
    mb.class_('::cv::Point_<int>').rename('Point2i')
    zz = mb.classes(lambda z: z.name.startswith('Point_<'))
    for z in zz:
        z.include()
        z.operator(lambda x: x.name.endswith('::CvPoint')).rename('as_CvPoint')
        z.operator(lambda x: x.name.endswith('::CvPoint2D32f')).rename('as_CvPoint2D32f')
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
    
    # Point3 et al
    mb.class_('::cv::Point3_<float>').rename('Point3f')
    zz = mb.classes(lambda z: z.name.startswith('Point3_<'))
    for z in zz:
        z.include()
        z.operator(lambda x: x.name.endswith('::CvPoint3D32f')).rename('as_CvPoint3D32f')
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
    
    # Size et al
    mb.class_('::cv::Size_<int>').rename('Size')
    zz = mb.classes(lambda z: z.name.startswith('Size_<'))
    for z in zz:
        z.include()
        z.operator(lambda x: x.name.endswith('::CvSize')).rename('as_CvSize')
        z.operator(lambda x: x.name.endswith('::CvSize2D32f')).rename('as_CvSize2D32f')
    
    # Rect et al
    zz = mb.classes(lambda z: z.name.startswith('Rect_<'))
    for z in zz:
        z.include()
        z.operator(lambda x: x.name.endswith('::CvRect')).rename('as_CvRect')
    
    # RotatedRect
    z = mb.class_('RotatedRect')
    z.include()
    z.operator(lambda x: x.name.endswith('::CvBox2D')).rename('as_CvBox2D')
    
    # Scalar et al
    # TODO: provide interface to ndarray
    mb.class_('::cv::Scalar_<double>').rename('Scalar')
    zz = mb.classes(lambda z: z.name.startswith('Scalar_<'))
    for z in zz:
        z.include()
        z.operator(lambda x: x.name.endswith('::CvScalar')).rename('as_CvScalar')
    
    # Range
    z = mb.class_('Range')
    z.include()
    z.operator(lambda x: x.name.endswith('::CvSlice')).rename('as_CvSlice')
    
    # Mat
    # TODO: provide interface to ndarray
    z = mb.class_('Mat')
    z.include()
    z.constructor(lambda x: '::IplImage' in x.decl_string).exclude()
    z.constructor(lambda x: '::CvMat' in x.decl_string).exclude()
    z.operator(lambda x: x.name.endswith('::CvMat')).rename('as_CvMat')
    z.operator(lambda x: x.name.endswith('::IplImage')).rename('as_IplImage')
    z.decls(lambda x: 'MatExpr' in x.decl_string).exclude()
    z.mem_funs('setTo').call_policies = CP.return_self()
    z.mem_funs('adjustROI').call_policies = CP.return_self()
    for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        z.decls(t).exclude()

    # RNG
    z = mb.class_('RNG')
    z.include()
    z.operator(lambda x: x.name.endswith('uchar')).rename('as_uchar')
    z.operator(lambda x: x.name.endswith('schar')).rename('as_schar')
    z.operator(lambda x: x.name.endswith('ushort')).rename('as_ushort')
    z.operator(lambda x: x.name.endswith('short int')).rename('as_short')
    z.operator(lambda x: x.name.endswith('unsigned int')).rename('as_unsigned')
    z.operator(lambda x: x.name.endswith('operator int')).rename('as_int')
    z.operator(lambda x: x.name.endswith('float')).rename('as_float')
    z.operator(lambda x: x.name.endswith('double')).rename('as_double')
    
    # TermCriteria
    z = mb.class_('TermCriteria')
    z.include()
    z.operator(lambda x: x.name.endswith('CvTermCriteria')).rename('as_CvTermCriteria')
    
    # PCA and SVD
    for t in ('::cv::PCA', '::cv::SVD'):
        z = mb.class_(t)
        z.include()
        z.operator('()').call_policies = CP.return_self()
        
    # LineIterator
    z = mb.class_('LineIterator')
    z.include()
    z.decls(lambda x: 'uchar *' in x.decl_string).exclude()
    
    # MatND
    # TODO: provide interface to ndarray
    # TODO: fix the rest of the member declarations
    z = mb.class_('MatND')
    z.include()
    z.decls().exclude()

    # z.constructor(lambda x: '::CvMatND' in x.decl_string).exclude()
    # z.operator(lambda x: x.name.endswith('::CvMatND')).rename('as_CvMatND')
    
    # z.decls(lambda x: 'Range*' in x.decl_string).exclude()
    
    # z.mem_funs('setTo').call_policies = CP.return_self()
    # z.mem_funs('adjustROI').call_policies = CP.return_self()
    # for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        # z.decls(t).exclude()

    # NAryMatNDIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('NAryMatNDIterator')
    z.include()
    z.decls().exclude()
    
    # SparseMat
    # TODO: fix the rest of the member declarations
    z = mb.class_('SparseMat')
    z.include()
    z.decls().exclude()
    
    # SparseMatConstIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('SparseMatConstIterator')
    z.include()
    z.decls().exclude()
    
    # SparseMatIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('SparseMatIterator')
    z.include()
    z.decls().exclude()
    
    # KDTree
    # TODO: fix the rest of the member declarations
    z = mb.class_('KDTree')
    z.include()
    z.decls().exclude()
    
    # FileStorage
    # TODO: fix the rest of the member declarations
    z = mb.class_('FileStorage')
    z.include()
    z.decls().exclude()
    
    # FileNode
    # TODO: fix the rest of the member declarations
    z = mb.class_('FileNode')
    z.include()
    z.decls().exclude()
    
    # FileNodeIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('FileNodeIterator')
    z.include()
    z.decls().exclude()
    
    
    #=============================================================================
    # Structures
    #=============================================================================
    

    # free functions
    for z in ('fromUtf16', 'toUtf16',
        'setNumThreads', 'getNumThreads', 'getThreadNum',
        'getTickCount', 'getTickFrequency',
        'setUseOptimized', 'useOptimized',
        ):
        mb.free_fun(lambda decl: z in decl.name).include()
        
    # free functions
    for z in (
        'getElemSize',
        # 'cvarrToMat', 'extractImageCOI', 'insertImageCOI', # removed, everything is in ndarray now
        'add', 'subtract', 'multiply', 'divide', 'scaleAdd', 'addWeighted',
        'convertScaleAbs', 'LUT', 'sum', 'countNonZero', 'mean', 'meanStdDev', 
        'normalize', 'reduce', 'flip', 'repeat', 'bitwise_and', 'bitwise_or', 
        'bitwise_xor', 'bitwise_not', 'absdiff', 'inRange', 'compare', 'min', 
        'max', 'sqrt', 'pow', 'exp', 'log', 'cubeRoot', 'fastAtan2',
        'polarToCart', 'cartToPolar', 'phase', 'magnitude', 'gemm',
        'mulTransposed', 'transpose', 'transform', 'perspectiveTransform',
        'completeSymm', 'setIdentity', 'determinant', 'trace', 'invert', 
        'solve', 'sort', 'sortIdx', 'eigen', 'Mahalanobis', 'Mahalonobis', 
        'dft', 'idft', 'dct', 'idct', 'mulSpectrums', 'getOptimalDFTSize',
        'randu', 'randn', 'line', 'rectangle', 'circle', 'ellipse', 'clipLine',
        'putText', 
        ):
        mb.free_funs(z).include()

    # TODO: 
    # minMaxLoc, merge, split, mixChannels, checkRange, calcCovarMatrix, kmeans, theRNG
    # randShuffle, fillConvexPoly, fillPoly, polylines, ellipse2Poly, getTextSize
    
    # TODO: missing functions; 'solveCubic', 'solvePoly', 

    # TODO: do something with Seq<>
