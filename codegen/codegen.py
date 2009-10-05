
import os
from pygccxml import declarations
from pyplusplus import module_builder, messages
import function_transformers as FT
from pyplusplus.module_builder import call_policies as CP

#Creating an instance of class that will help you to expose your declarations
mb = module_builder.module_builder_t( 
    [
        "cxcore.h", 
        # "cxcore.hpp", 
        # "cv.h", 
        # "cv.hpp", 
        # "cvaux.h", 
        # "cvaux.hpp", 
        # "ml.h", 
        # "highgui.h", 
        # "highgui.hpp",
        # r"M:/programming/mypackages/pyopencv/workspace_svn/pyopencv_opencv1.2b_win32/pyopencvext.hpp"
    ],
    gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe", 
    working_directory=r"M:/programming/mypackages/pyopencv/svn_workplace/trunk/codegen", 
    include_paths=[
        r"M:/programming/mypackages/pyopencv/svn_workplace/trunk/codegen/opencv2_include",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
    ],
    define_symbols=[] )
    
cc = open('pyopencv/__init__.py', 'w')
cc.write('''#!/usr/bin/env python
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

from pyopencvext import *
import pyopencvext as _PE
import math as _Math
import ctypes as _CT


#=============================================================================
# cvver.h
#=============================================================================

CV_MAJOR_VERSION    = 2
CV_MINOR_VERSION    = 0
CV_SUBMINOR_VERSION = 0
CV_VERSION          = "2.0.0"




''')



#=============================================================================
# Initialization
#=============================================================================


#Well, don't you want to see what is going on?
# mb.print_declarations() -- too many declarations

# Disable every declarations first
mb.decls().exclude()

# disable some warnings
mb.decls().disable_warnings(messages.W1027, messages.W1025)

# expose 'this'
mb.classes().expose_this = True



#=============================================================================
# Rules for free functions and member functions
#=============================================================================


# initialize list of transformer creators for each free function
for z in mb.free_funs():
    z._transformer_creators = []
for z in mb.mem_funs():
    z._transformer_creators = []
    
# by default, convert all pointers to Cv... or to Ipl... into pointee
for z in mb.free_funs():
    for arg in z.arguments:
        if declarations.is_pointer(arg.type) and not declarations.is_pointer(declarations.remove_pointer(arg.type)) and \
            (arg.type.decl_string.startswith('::Cv') or arg.type.decl_string.startswith('::_Ipl')):
            z._transformer_creators.append(FT.input_smart_pointee(arg.name))
# for z in mb.mem_funs(): # TODO: fix
    # for i in xrange(z.arguments):
        # if declarations.is_pointer(z.arguments[i]):
            # z._transformer_creators.append(FT.input_smart_pointee(i+1))


cc.write('''
#=============================================================================
# cxerror.h
#=============================================================================


''')

# CVStatus
mb.decl('CVStatus').include()
cc.write('''
CV_StsOk                   =  0  # everithing is ok                
CV_StsBackTrace            = -1  # pseudo error for back trace     
CV_StsError                = -2  # unknown /unspecified error      
CV_StsInternal             = -3  # internal error (bad state)      
CV_StsNoMem                = -4  # insufficient memory             
CV_StsBadArg               = -5  # function arg/param is bad       
CV_StsBadFunc              = -6  # unsupported function            
CV_StsNoConv               = -7  # iter. didn't converge           
CV_StsAutoTrace            = -8  # tracing                         

CV_HeaderIsNull            = -9  # image header is NULL            
CV_BadImageSize            = -10 # image size is invalid           
CV_BadOffset               = -11 # offset is invalid               
CV_BadDataPtr              = -12 #
CV_BadStep                 = -13 #
CV_BadModelOrChSeq         = -14 #
CV_BadNumChannels          = -15 #
CV_BadNumChannel1U         = -16 #
CV_BadDepth                = -17 #
CV_BadAlphaChannel         = -18 #
CV_BadOrder                = -19 #
CV_BadOrigin               = -20 #
CV_BadAlign                = -21 #
CV_BadCallBack             = -22 #
CV_BadTileSize             = -23 #
CV_BadCOI                  = -24 #
CV_BadROISize              = -25 #

CV_MaskIsTiled             = -26 #

CV_StsNullPtr                = -27 # null pointer 
CV_StsVecLengthErr           = -28 # incorrect vector length 
CV_StsFilterStructContentErr = -29 # incorr. filter structure content 
CV_StsKernelStructContentErr = -30 # incorr. transform kernel content 
CV_StsFilterOffsetErr        = -31 # incorrect filter ofset value 

#extra for CV 
CV_StsBadSize                = -201 # the input/output structure size is incorrect  
CV_StsDivByZero              = -202 # division by zero 
CV_StsInplaceNotSupported    = -203 # in-place operation is not supported 
CV_StsObjectNotFound         = -204 # request can't be completed 
CV_StsUnmatchedFormats       = -205 # formats of input/output arrays differ 
CV_StsBadFlag                = -206 # flag is wrong or not supported   
CV_StsBadPoint               = -207 # bad CvPoint  
CV_StsBadMask                = -208 # bad format of mask (neither 8uC1 nor 8sC1)
CV_StsUnmatchedSizes         = -209 # sizes of input/output structures do not match 
CV_StsUnsupportedFormat      = -210 # the data format/type is not supported by the function
CV_StsOutOfRange             = -211 # some of parameters are out of range 
CV_StsParseError             = -212 # invalid syntax/structure of the parsed file 
CV_StsNotImplemented         = -213 # the requested function/feature is not implemented 
CV_StsBadMemBlock            = -214 # an allocated block has been corrupted 
CV_StsAssert                 = -215 # assertion failed 



''')



cc.write('''
#=============================================================================
# cxtypes.h
#=============================================================================


''')

# CvArr
mb.class_('CvArr').include()

cc.write('''
#-----------------------------------------------------------------------------
# Common macros and inline functions
#-----------------------------------------------------------------------------

CV_PI = _Math.pi
CV_LOG2 = 0.69314718055994530941723212145818

''')

# functions
for z in ('cvRound', 'cvFloor', 'cvCeil', 'cvIsNaN', 'cvIsInf'):
    mb.free_fun(z).include()

cc.write('''
#-----------------------------------------------------------------------------
# Random number generation
#-----------------------------------------------------------------------------

# Minh-Tri's note: I'd rather use a random generator other than CvRNG.
# It's slow and doesn't guarrantee a large cycle.

CvRNG = _CT.c_uint64

def cvRNG(seed=-1):
    """CvRNG cvRNG( int64 seed = CV_DEFAULT(-1))
    
    Initializes random number generator and returns the state. 
    """
    if seed != 0:
        return CvRNG(seed)
    return CvRNG(-1)

def cvRandInt(rng):
    """unsigned cvRandInt( CvRNG rng )
    
    Returns random 32-bit unsigned integer. 
    """
    temp = rng.value
    temp = temp*4164903690 + (temp >> 32)
    rng.value = temp
    return _CT.c_uint32(temp).value
    
def cvRandReal(rng):
    """double cvRandReal( CvRNG rng )
    
    Returns random floating-point number between 0 and 1.
    """
    return _CT.c_double(cvRandInt(rng).value*2.3283064365386962890625e-10) # 2^-32

    
''')


# IplImage and IplROI
cc.write('''
#-----------------------------------------------------------------------------
# Image type (IplImage)
#-----------------------------------------------------------------------------

# Image type (IplImage)
IPL_DEPTH_SIGN = -0x80000000

IPL_DEPTH_1U =  1
IPL_DEPTH_8U =  8
IPL_DEPTH_16U = 16
IPL_DEPTH_32F = 32
IPL_DEPTH_64F = 64

IPL_DEPTH_8S = IPL_DEPTH_SIGN + IPL_DEPTH_8U
IPL_DEPTH_16S = IPL_DEPTH_SIGN + IPL_DEPTH_16U
IPL_DEPTH_32S = IPL_DEPTH_SIGN + 32

IPL_DATA_ORDER_PIXEL = 0
IPL_DATA_ORDER_PLANE = 1

IPL_ORIGIN_TL = 0
IPL_ORIGIN_BL = 1

IPL_ALIGN_4BYTES = 4
IPL_ALIGN_8BYTES = 8
IPL_ALIGN_16BYTES = 16
IPL_ALIGN_32BYTES = 32

IPL_ALIGN_DWORD = IPL_ALIGN_4BYTES
IPL_ALIGN_QWORD = IPL_ALIGN_8BYTES

IPL_BORDER_CONSTANT = 0
IPL_BORDER_REPLICATE = 1
IPL_BORDER_REFLECT = 2
IPL_BORDER_WRAP = 3

IPL_IMAGE_HEADER = 1
IPL_IMAGE_DATA = 2
IPL_IMAGE_ROI = 4

IPL_BORDER_REFLECT_101    = 4

CV_TYPE_NAME_IMAGE = "opencv-image"


''')    

iplimage = mb.class_('_IplImage')
iplimage.rename('IplImage')
iplimage.include()
for z in ('imageId', 'imageData', 'imageDataOrigin', 'tileInfo', 'maskROI'): # don't need these attributes
    iplimage.var(z).exclude()
FT.expose_member_as_pointee(iplimage, 'roi')    
# deal with 'imageData' and 'roi'
iplimage.include_files.append( "boost/python/object.hpp" )
iplimage.include_files.append( "boost/python/str.hpp" )
iplimage.add_wrapper_code('''

    static bp::object get_data( _IplImage const & inst ){        
        return inst.imageData? bp::str(inst.imageData, inst.imageSize) : bp::object();
    }

''')
iplimage.add_registration_code('''
add_property( "data", bp::make_function(&_IplImage_wrapper::get_data) )
''')

cc.write('''
IplImage._owner = 0 # default: owns nothing
        
def _IplImage__del__(self):
    if self._owner == 1: # own header only
        _PE._cvReleaseImageHeader(self)
    elif self._owner == 2: # own data but not header
        _PE._cvReleaseData(self)
    elif self._owner == 3: # own header and data
        _PE._cvReleaseImage(self)
IplImage.__del__ = _IplImage__del__

''')

# IplROI
z = mb.class_('_IplROI')
z.rename('IplROI')
z.include()

# IplConvKernel
z = mb.class_('_IplConvKernel')
z.rename('IplConvKernel')
z.include()
z.var('values').exclude() # don't need this variable yet

cc.write('''
IplConvKernel._owner = False # default: owns nothing
        
def _IplConvKernel__del__(self):
    if self._owner is True: # own header
        _PE._cvReleaseStructuringElement(self)
IplConvKernel.__del__ = _IplConvKernel__del__

''')


# CvMat
cc.write('''
#-----------------------------------------------------------------------------
# Matrix type (CvMat) 
#-----------------------------------------------------------------------------

# Matrix type (CvMat)
CV_CN_MAX = 64
CV_CN_SHIFT = 3
CV_DEPTH_MAX = (1 << CV_CN_SHIFT)

CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6
CV_USRTYPE1 = 7

def CV_MAKETYPE(depth,cn):
    return ((depth) + (((cn)-1) << CV_CN_SHIFT))
CV_MAKE_TYPE = CV_MAKETYPE

CV_8UC1 = CV_MAKETYPE(CV_8U,1)
CV_8UC2 = CV_MAKETYPE(CV_8U,2)
CV_8UC3 = CV_MAKETYPE(CV_8U,3)
CV_8UC4 = CV_MAKETYPE(CV_8U,4)

CV_8SC1 = CV_MAKETYPE(CV_8S,1)
CV_8SC2 = CV_MAKETYPE(CV_8S,2)
CV_8SC3 = CV_MAKETYPE(CV_8S,3)
CV_8SC4 = CV_MAKETYPE(CV_8S,4)

CV_16UC1 = CV_MAKETYPE(CV_16U,1)
CV_16UC2 = CV_MAKETYPE(CV_16U,2)
CV_16UC3 = CV_MAKETYPE(CV_16U,3)
CV_16UC4 = CV_MAKETYPE(CV_16U,4)

CV_16SC1 = CV_MAKETYPE(CV_16S,1)
CV_16SC2 = CV_MAKETYPE(CV_16S,2)
CV_16SC3 = CV_MAKETYPE(CV_16S,3)
CV_16SC4 = CV_MAKETYPE(CV_16S,4)

CV_32SC1 = CV_MAKETYPE(CV_32S,1)
CV_32SC2 = CV_MAKETYPE(CV_32S,2)
CV_32SC3 = CV_MAKETYPE(CV_32S,3)
CV_32SC4 = CV_MAKETYPE(CV_32S,4)

CV_32FC1 = CV_MAKETYPE(CV_32F,1)
CV_32FC2 = CV_MAKETYPE(CV_32F,2)
CV_32FC3 = CV_MAKETYPE(CV_32F,3)
CV_32FC4 = CV_MAKETYPE(CV_32F,4)

CV_64FC1 = CV_MAKETYPE(CV_64F,1)
CV_64FC2 = CV_MAKETYPE(CV_64F,2)
CV_64FC3 = CV_MAKETYPE(CV_64F,3)
CV_64FC4 = CV_MAKETYPE(CV_64F,4)

CV_AUTOSTEP = 0x7fffffff
CV_WHOLE_ARR  = cvSlice( 0, 0x3fffffff )

CV_MAT_CN_MASK = ((CV_CN_MAX - 1) << CV_CN_SHIFT)
def CV_MAT_CN(flags):
    return ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
CV_MAT_DEPTH_MASK = (CV_DEPTH_MAX - 1)
def CV_MAT_DEPTH(flags):
    return ((flags) & CV_MAT_DEPTH_MASK)
CV_MAT_TYPE_MASK = (CV_DEPTH_MAX*CV_CN_MAX - 1)
def CV_MAT_TYPE(flags):
    ((flags) & CV_MAT_TYPE_MASK)
CV_MAT_CONT_FLAG_SHIFT = 9
CV_MAT_CONT_FLAG = (1 << CV_MAT_CONT_FLAG_SHIFT)
def CV_IS_MAT_CONT(flags):
    return ((flags) & CV_MAT_CONT_FLAG)
CV_IS_CONT_MAT = CV_IS_MAT_CONT
CV_MAT_TEMP_FLAG_SHIFT = 10
CV_MAT_TEMP_FLAG = (1 << CV_MAT_TEMP_FLAG_SHIFT)
def CV_IS_TEMP_MAT(flags):
    return ((flags) & CV_MAT_TEMP_FLAG)

CV_MAGIC_MASK = 0xFFFF0000
CV_MAT_MAGIC_VAL = 0x42420000
CV_TYPE_NAME_MAT = "opencv-matrix"


''')

cvmat = mb.class_('CvMat')
cvmat.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmat.var(z).exclude()
# deal with 'data'
cvmat.include_files.append( "boost/python/object.hpp" )
cvmat.include_files.append( "boost/python/str.hpp" )
cvmat.add_wrapper_code('''
static bp::object get_data( CvMat const & inst ){        
    return inst.data.ptr? bp::str((const char *)inst.data.ptr, (inst.step? inst.step: CV_ELEM_SIZE(inst.type)*inst.cols)*inst.rows) : bp::object();
}
''')
cvmat.add_registration_code('''
add_property( "data", bp::make_function(&CvMat_wrapper::get_data) )
''')

cc.write('''
CvMat._owner = False
        
def _CvMat__del__(self):
    if self._owner is True:
        _PE._cvReleaseMat(self)
CvMat.__del__ = _CvMat__del__

''')

# CvMat functions
for z in ('cvMat', 'cvmGet', 'cvmSet', 'cvIplDepth'):
    mb.free_fun(z).include()


# CvMatND
cc.write('''
#-----------------------------------------------------------------------------
# Multi-dimensional dense array (CvMatND)
#-----------------------------------------------------------------------------

CV_MATND_MAGIC_VAL    = 0x42430000
CV_TYPE_NAME_MATND    = "opencv-nd-matrix"

CV_MAX_DIM = 32
CV_MAX_DIM_HEAP = (1 << 16)


''')

cvmatnd = mb.class_('CvMatND')
cvmatnd.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmatnd.var(z).exclude()
# deal with 'data'
cvmatnd.include_files.append( "boost/python/object.hpp" )
cvmatnd.include_files.append( "boost/python/str.hpp" )
cvmatnd.add_wrapper_code('''
static bp::object get_data( CvMatND const & inst ){        
    return inst.data.ptr? bp::str((const char *)inst.data.ptr, inst.dim[0].step*inst.dim[0].size): bp::object();
}
''')
cvmatnd.add_registration_code('''
add_property( "data", bp::make_function(&CvMatND_wrapper::get_data) )
''')

cc.write('''
CvMatND._owner = False
        
def _CvMatND__del__(self):
    if self._owner is True:
        _PE._cvReleaseMat(self)
CvMatND.__del__ = _CvMatND__del__

''')


# CvSparseMat
cc.write('''
#-----------------------------------------------------------------------------
# Multi-dimensional sparse array (CvSparseMat) 
#-----------------------------------------------------------------------------

CV_SPARSE_MAT_MAGIC_VAL    = 0x42440000
CV_TYPE_NAME_SPARSE_MAT    = "opencv-sparse-matrix"


''')

cvsparsemat = mb.class_('CvSparseMat')
cvsparsemat.include()
for z in ('heap', 'hashtable'): # TODO: fix
    cvsparsemat.var(z).exclude()

cc.write('''
CvSparseMat._owner = False
        
def _CvSparseMat__del__(self):
    _PE._cvReleaseSparseMat(self)
CvSparseMat.__del__ = _CvSparseMat__del__

''')


# CvSparseNode
z = mb.class_('CvSparseNode')
z.include()
FT.expose_member_as_pointee(z, 'next')

# CvSparseMatIterator
z = mb.class_('CvSparseMatIterator')
z.include()
FT.expose_member_as_pointee(z, 'mat')
FT.expose_member_as_pointee(z, 'node')


# CvHistogram
cvhistogram = mb.class_('CvHistogram')
cvhistogram.include()
for z in ('bins', 'thresh', 'thresh2'): # TODO: fix this
    cvhistogram.var(z).exclude()
mb.decl('CvHistType').include()
cc.write('''
#-----------------------------------------------------------------------------
# Histogram
#-----------------------------------------------------------------------------

CV_HIST_MAGIC_VAL     = 0x42450000
CV_HIST_UNIFORM_FLAG  = (1 << 10)

CV_HIST_RANGES_FLAG   = (1 << 11)

CV_HIST_ARRAY         = 0
CV_HIST_SPARSE        = 1
CV_HIST_TREE          = CV_HIST_SPARSE

CV_HIST_UNIFORM       = 1


''')

cc.write('''
CvHistogram._owner = False
        
def _CvHistogram__del__(self):
    if self._owner is True:
        _PE._cvReleaseHist(self)
CvHistogram.__del__ = _CvHistogram__del__

''')


# Other supplementary data type definitions
cc.write('''
#-----------------------------------------------------------------------------
# Other supplementary data type definitions
#-----------------------------------------------------------------------------

CV_TERMCRIT_ITER    = 1
CV_TERMCRIT_NUMBER  = CV_TERMCRIT_ITER
CV_TERMCRIT_EPS     = 2

CV_WHOLE_SEQ_END_INDEX = 0x3fffffff
CV_WHOLE_SEQ = cvSlice(0, CV_WHOLE_SEQ_END_INDEX)


''')

for z in ('CvRect',):
    mb.class_(z).include()
for z in ('cvScalar', 'cvScalarAll', 
    'cvRect', 'cvRectToROI', 'cvROIToRect'):
    mb.free_fun(z).include()
for z in ('CvScalar', 'cvRealScalar', 
    'CvPoint', 'cvPoint', 
    'CvSize', 'cvSize', 'CvBox2D',
    'CvTermCriteria', 'cvTermCriteria', 'cvCheckTermCriteria',
    'CvLineIterator',
    'CvSlice', 'cvSlice',
    ):
    mb.decls(lambda decl: decl.name.startswith(z)).include()
    
mb.class_('CvLineIterator').var('ptr').exclude() # TODO: fix this
    

# Dynamic Data structures
cc.write('''
#-----------------------------------------------------------------------------
# Dynamic Data structures
#-----------------------------------------------------------------------------

CV_STORAGE_MAGIC_VAL = 0x42890000

CV_TYPE_NAME_SEQ             = "opencv-sequence"
CV_TYPE_NAME_SEQ_TREE        = "opencv-sequence-tree"

CV_SET_ELEM_IDX_MASK   = ((1 << 26) - 1)
CV_SET_ELEM_FREE_FLAG  = (1 << (_CT.sizeof(_CT.c_int)*8-1))

# Checks whether the element pointed by ptr belongs to a set or not
def CV_IS_SET_ELEM(ptr):
    return cast(ptr, CvSetElem_p)[0].flags >= 0


''')

# CvMemBlock
z = mb.class_('CvMemBlock')
z.include()
for t in ('prev', 'next'):
    FT.expose_member_as_pointee(z, t)

# CvMemStorage
z = mb.class_('CvMemStorage')
z.include()
for t in ('bottom', 'top', 'parent'):
    FT.expose_member_as_pointee(z, t)

# CvMemStoragePos
z = mb.class_('CvMemStoragePos')
z.include()
FT.expose_member_as_pointee(z, 'top')

# CvSeqBlock
z = mb.class_('CvSeqBlock')
z.include()
for t in ('prev', 'next'):
    FT.expose_member_as_pointee(z, t)
FT.expose_member_as_str(z, 'data')

# CvSeq
z = mb.class_('CvSeq')
def _expose_CvSeq_members(z):
    z.include()
    for t in ('h_prev', 'h_next', 'v_prev', 'v_next', 'storage', 'free_blocks', 'first'):
        FT.expose_member_as_pointee(z, t)
    for t in ('block_max', 'ptr'):
        FT.expose_member_as_str(z, t)
_expose_CvSeq_members(z)
        
# CvSetElem
z = mb.class_('CvSetElem')
z.include()
FT.expose_member_as_pointee(z, 'next_free')

# CvSet
z = mb.class_('CvSet')
def _expose_CvSet_members(z):
    _expose_CvSeq_members(z)
    FT.expose_member_as_pointee(z, 'free_elems')
_expose_CvSet_members(z)





cc.write('''
#=============================================================================
# CxCore/Basic Structures
#=============================================================================


''')

    
   

def add_underscore(decl):
    decl.rename('_'+decl.name)
    decl.include()


cc.write('''
#=============================================================================
# CxCore/Operations on Arrays
#=============================================================================


''')

# return_pointee_value call policies for cvCreate... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvCreate')):
    if not z.name in ('cvCreateData', 'cvCreateSeqBlock'): # TODO: fix
        add_underscore(z)
        z.call_policies = CP.return_value_policy( CP.return_pointee_value )

# return_pointee_value call policies for cvClone... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvClone')):
    if not z.name in ('cvClone', ):
        add_underscore(z)
        z.call_policies = CP.return_value_policy( CP.return_pointee_value )

# cvRelease... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvRelease')):
    if not z.name in ('cvRelease', 'cvReleaseMemStorage', 'cvReleaseData', 'cvReleaseFileStorage'): # TODO: fix
        add_underscore(z)
        z._transformer_creators.append(FT.input_double_pointee(0))
        
# cvInit... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvInit')):
    add_underscore(z)
    z.call_policies = CP.return_self()

# function argument 'void *data'
for f in mb.free_funs():
    for arg in f.arguments:
        if arg.name == 'data' and 'void *' in arg.type.decl_string:
            f._transformer_creators.append(FT.input_string(arg.name))
            break

# cvCreateImage
cc.write('''
def cvCreateImage(size, depth, channels):
    """IplImage cvCreateImage(CvSize size, int depth, int channels)

    Creates header and allocates data
    """
    z = _PE._cvCreateImage(size, depth, channels)
    if z is not None:
        z._owner = 3 # both header and data
    return z

''')

        
# cvCreateImageHeader
cc.write('''
def cvCreateImageHeader(size, depth, channels):
    """IplImage cvCreateImageHeader(CvSize size, int depth, int channels)

    Allocates, initializes, and returns structure IplImage
    """
    z = _PE._cvCreateImageHeader(size, depth, channels)
    if z is not None:
        z._owner = 1 # header only
    return z

''')

# cvInitImageHeader
cc.write('''
def cvInitImageHeader(image, size, depth, channels, origin=0, align=4):
    """IplImage cvInitImageHeader(IplImage image, CvSize size, int depth, int channels, int origin=0, int align=4)

    Initializes allocated by user image header
    """
    _PE._cvInitImageHeader(image, size, depth, channels, origin=origin, align=align)
    return image
    
''')

# cvCloneImage
cc.write('''
def cvCloneImage(image):
    """IplImage cvCloneImage(const IplImage image)

    Makes a full copy of image (widthStep may differ)
    """
    z = _PE._cvCloneImage(image)
    if z is not None:
        z._owner = 3 # as a clone, z owns both header and data
    return z

''')


# cv...ImageCOI functions
mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageCOI' in decl.name).include()

# cv...ImageROI functions
mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageROI' in decl.name).include()



# cvCreateMat
cc.write('''
def cvCreateMat(rows, cols, cvmat_type):
    """CvMat cvCreateMat(int rows, int cols, int type)

    Creates new matrix
    """
    z = _PE._cvCreateMat(rows, cols, cvmat_type)
    if z is not None:
        z._owner = True
    return z

''')

cc.write('''
# Minh-Tri's helpers
def cvCreateMatFromCvPoint2D32fList(points):
    """CvMat cvCreateMatFromCvPoint2D32fList(list_or_tuple_of_CvPoint2D32f points)
    
    Creates a new matrix from a list/tuple of CvPoint2D32f points
    """
    cols = len(points)
    z = cvCreateMat(1, cols, CV_32FC2)
    for i in range(cols):
        x = points[i]
        y = z[0,i]
        y[0] = x.x
        y[1] = x.y
    return z

def cvCreateMatFromCvPointList(points):
    """CvMat cvCreateMatFromCvPointList(list_or_tuple_of_CvPoint points)
    
    Creates a new matrix from a list/tuple of CvPoint points
    """
    cols = len(points)
    z = cvCreateMat(1, cols, CV_32SC2)
    for i in range(cols):
        x = points[i]
        y = z[0,i]
        y[0] = x.x
        y[1] = x.y
    return z

''')

# cvCreateMatHeader
cc.write('''
def cvCreateMatHeader(rows, cols, cvmat_type):
    """CvMat cvCreateMatHeader(int rows, int cols, int type)

    Creates new matrix header
    """
    z = _PE._cvCreateMatHeader(rows, cols, cvmat_type)
    if z is not None:
        z._owner = True
    return z

CV_AUTOSTEP = 0x7fffffff

''')


# cvInitMatHeader
cc.write('''
def cvInitMatHeader(mat, rows, cols, cvmat_type, data=None, step=CV_AUTOSTEP):
    """CvMat cvInitMatHeader(CvMat mat, int rows, int cols, int type, string data=None, int step=CV_AUTOSTEP)

    Initializes matrix header
    """
    _PE._cvInitMatHeader(mat, rows, cols, cvmat_type, data=data, step=step)
    if data is not None:
        mat._depends = (data,)
    return mat
    
''')


# cvCloneMat
cc.write('''
def cvCloneMat(mat):
    """CvMat cvCloneMat(const CvMat mat)

    Creates matrix copy
    """
    z = _PE._cvCloneMat(mat)
    if z is not None:
        z._owner = True
    return z

''')

# cvReleaseData
add_underscore(mb.free_fun('cvReleaseData'))

# -----------------------------------------------------------------------------------------------
# Final tasks
# -----------------------------------------------------------------------------------------------

for z in ('hdr_refcount', 'refcount'): # too low-level
    mb.decls(z).exclude() 

# mb.free_function( return_type='IplImage *' ).call_policies \
    # = call_policies.return_value_policy( call_policies.return_pointee_value )
    
# apply all the function transformations    
for z in mb.free_funs():
    if len(z._transformer_creators) > 0:
        z.add_transformation(*z._transformer_creators)


for z in ('IPL_', 'CV_'):
    try:
        mb.decls(lambda decl: decl.name.startswith(z)).include()
    except RuntimeError:
        pass













# exlude every class first
# mb.classes().exclude()

# expose every OpenCV's C structure and class but none of its members
# for z in mb.classes(lambda z: z.decl_string.startswith('::Cv') or z.decl_string.startswith('::_Ipl')):
    # z.include()
    # z.decls().exclude()
    
# exclude stupid CvMat... aliases
# mb.classes(lambda z: z.decl_string.startswith('::CvMat') and not z.name.startswith('CvMat')).exclude()
    
# cannot expose unions
# mb.class_('Cv32suf').exclude()
# mb.class_('Cv64suf').exclude()

# expose every OpenCV's C++ class but none of its members
# for z in mb.classes(lambda z: z.decl_string.startswith('::cv')):
    # z.include()
    # z.decls().exclude()
    
# exclude every Ptr class
# mb.classes(lambda z: z.decl_string.startswith('::cv::Ptr')).exclude()

# exclude every MatExpr class
# mb.classes(lambda z: z.decl_string.startswith('::cv::MatExpr')).exclude()

# expose every OpenCV's C++ free function
# mb.free_functions(lambda z: z.decl_string.startswith('::cv')).include()

# -----------------------------------------------------------------------------------------------
# cxtypes.h
# -----------------------------------------------------------------------------------------------

# CvTypeInfo
# cvtypeinfo = mb.class_('CvTypeInfo')
# expose_member_as_str(cvtypeinfo, 'type_name')
# for z in ('is_instance', 'release', 'read', 'write', 'clone'):
    # expose_addressof_member(cvtypeinfo, z)
    
# CvAttrList


# -----------------------------------------------------------------------------------------------

# for z in ('_IplImage', 'CvAttrList', 'CvFileNode', 'CvMatND', '_IplConvKernelFP', 
    # 'CvModuleInfo', 'CvChain', 'CvHistogram', 'CvSeqReader', 'CvContour',
    # 'CvString', 'CvSet', 'CvGraph', 'CvSeqWriter', 'CvSeq', 'CvSeqBlock', 'CvGraphEdge',
    # '_IplConvKernel', 'CvPluginFuncInfo', 'CvLineIterator', 'CvSparseMat', 'CvString',
    # '_IplROI', ):
    # mb.class_(z).exclude()
    
    
# cv = mb.namespace('cv')
# cv.decls().exclude()

# cv.decls(lambda decl: 'Optimized' in decl.name).include()

# for z in ('CvScalar', 'CvPoint', 'CvSize', 'CvRect', 'CvBox', 'CvSlice'):
    # mb.decls(lambda decl: decl.name.startswith(z)).include()


# -----------------------------------------------------------------------------------------------
# cxcore.hpp
# -----------------------------------------------------------------------------------------------
# cv = mb.namespace('cv') # namespace cv

# cv.class_('Exception').include()

# for z in ('Optimized', 'NumThreads', 'ThreadNum', 'getTick'):
    # cv.decls(lambda decl: z in decl.name).include()

# for z in ('DataDepth', 'Vec', 'Complex', 'Point', 'Size', 'Rect', 'RotatedRect', 
    # 'Scalar', 'Range', 'DataType'):
    # cv.decls(lambda decl: decl.name.startswith(z)).include()

# class Mat    
# mat = cv.class_('Mat')
# mat.include()
# for z in ('refcount', 'datastart', 'dataend'):
    # mat.var(z).exclude()
# TODO: expose the 'data' member as read-write buffer
# mat.var('data').exclude()
# expose_addressof_member(mat, 'data')    
# mat.decls('ptr').exclude()

#Creating code creator. After this step you should not modify/customize declarations.
mb.build_code_creator( module_name='pyopencvext' )

#Writing code to file.
mb.split_module( 'code' )
