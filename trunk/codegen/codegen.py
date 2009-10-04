
import os
from pygccxml import declarations
from pyplusplus import module_builder, messages
from pyplusplus import function_transformers as FT
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


#=============================================================================
# cvver.h
#=============================================================================

CV_MAJOR_VERSION    = 2
CV_MINOR_VERSION    = 0
CV_SUBMINOR_VERSION = 0
CV_VERSION          = "2.0.0"




''')


#Well, don't you want to see what is going on?
# mb.print_declarations() -- too many declarations

# Disable every declarations first
mb.decls().exclude()


def expose_addressof_member(klass, member_name, exclude_member=True):
    klass.include_files.append( "boost/python/long.hpp" )
    if exclude_member:
        klass.var(member_name).exclude()
    klass.add_declaration_code('''
    boost::python::long_ get_MEMBER_NAME_addr( CLASS_TYPE const & inst ){
        return boost::python::long_((int)&inst.MEMBER_NAME);
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('def( "get_MEMBER_NAME_addr", &::get_MEMBER_NAME_addr )'.replace("MEMBER_NAME", member_name))
    # TODO: finish the wrapping with ctypes code
    
def expose_member_as_str(klass, member_name):
    klass.include_files.append( "boost/python/object.hpp" )
    klass.include_files.append( "boost/python/str.hpp" )
    klass.var(member_name).exclude()
    klass.add_wrapper_code('''
    static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){        
        return inst.MEMBER_NAME? bp::str(inst.MEMBER_NAME): bp::object();
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    

def remove_ptr( type_ ):
    if declarations.is_pointer( type_ ):
        return declarations.remove_pointer( type_ )
    else:
        raise TypeError( 'Type should be a pointer, got %s.' % type_ )


# -----------------------------------------------------------------------------------------------
# Function transfomers
# -----------------------------------------------------------------------------------------------

# input_double_pointee_t
class input_double_pointee_t(FT.transformer_t):
    """Handles a double pointee input.
    
    Convert by dereferencing: do_smth(your_type **v) -> do_smth(your_type v)

    Right now compiler should be able to use implicit conversion
    """

    def __init__(self, function, arg_ref):
        FT.transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        if not declarations.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_double_pointee_t" transformation, argument %s type must be a pointer or a array (got %s).' ) \
                  % ( function, self.arg_ref.name, arg.type)

    def __str__(self):
        return "input_double_pointee(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        tmp_type = remove_ptr( self.arg.type )
        w_arg.type = remove_ptr( tmp_type )
        if not declarations.is_convertible( w_arg.type, self.arg.type ):
            controller.add_pre_call_code("%s tmp = reinterpret_cast< %s >(& %s);" % ( tmp_type, tmp_type, w_arg.name ))
            casting_code = 'reinterpret_cast< %s >( & tmp )' % self.arg.type
            controller.modify_arg_expression(self.arg_index, casting_code)

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return []

def input_double_pointee( *args, **keywd ):
    def creator( function ):
        return input_double_pointee_t( function, *args, **keywd )
    return creator

# input_smart_pointee_t
class input_smart_pointee_t(FT.transformer_t):
    """Handles a pointee input.
    
    Convert by dereferencing: do_smth(your_type *v) -> do_smth(object v2)
    where v2 is either of type NoneType or type 'your_type'. 
    If v2 is None, v is NULL.  Otherwise, v is the pointer to v2.
    """

    def __init__(self, function, arg_ref):
        FT.transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_smart_pointee(%s)" % self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ "boost/python/object.hpp", "boost/python/extract.hpp" ]

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        data_type = declarations.remove_const(remove_ptr( self.arg.type ))
        w_arg.type = declarations.dummy_type_t( "boost::python::object" )
        controller.add_pre_call_code("%s const &tmp = bp::extract<%s const &>(%s);" % (data_type, data_type, w_arg.name))
        controller.modify_arg_expression(self.arg_index, "(%s.ptr() != Py_None)? (%s)(&tmp): 0" % (w_arg.name, self.arg.type))

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )


def input_smart_pointee( *args, **keywd ):
    def creator( function ):
        return input_smart_pointee_t( function, *args, **keywd )
    return creator

# by default, convert all pointers into pointee
for z in mb.free_funs():
    for arg in z.arguments:
        if declarations.is_pointer(arg.type):
            if 'void' in arg.type.decl_string:
                z.add_transformation(FT.from_address(arg.name))
            else:
                z.add_transformation(input_smart_pointee(arg.name))
# for z in mb.mem_funs():
    # for i in xrange(z.arguments):
        # if declarations.is_pointer(z.arguments[i]):
            # z.add_transformation(input_smart_pointee(i+1))
    
    
#=============================================================================
# Initialization
#=============================================================================

# disable some warnings
mb.decls().disable_warnings(messages.W1027, messages.W1025)

# expose 'this'
mb.classes().expose_this = True




cc.write('''
#=============================================================================
# CxCore/Basic Structures
#=============================================================================


''')

cc.write('''
#-----------------------------------------------------------------------------
# CvTermCriteria
#-----------------------------------------------------------------------------

CV_TERMCRIT_ITER    = 1
CV_TERMCRIT_NUMBER  = CV_TERMCRIT_ITER
CV_TERMCRIT_EPS     = 2


''')

for z in ('CvScalar', 'cvRealScalar', 
    'CvPoint', 'cvPoint', 
    'CvSize', 'cvSize', 
    'CvTermCriteria', 'cvTermCriteria', 'cvCheckTermCriteria'):
    mb.decls(lambda decl: decl.name.startswith(z)).include()
for z in ('cvScalar', 'cvScalarAll', 
    'cvRect'):
    mb.decl(z).include()
mb.class_('CvRect').include()
    
cc.write('''
#-----------------------------------------------------------------------------
# Matrix type (CvMat) 
#-----------------------------------------------------------------------------

# Matrix type (CvMat)
CV_CN_MAX = 4
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

# CV_AUTO_STEP = 0x7fffffff -- no such thing as CV_AUTO_STEP
# CV_WHOLE_ARR  = cvSlice( 0, 0x3fffffff )

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


# CvMat
cvmat = mb.class_('CvMat')
cvmat.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmat.var(z).exclude()
# deal with 'data'
cvmat.include_files.append( "boost/python/object.hpp" )
cvmat.include_files.append( "boost/python/str.hpp" )
cvmat.add_wrapper_code('''
static bp::object get_data( CvMat const & inst ){        
    return bp::str((const char *)inst.data.ptr, (inst.step? inst.step: CV_ELEM_SIZE(inst.type)*inst.cols)*inst.rows);
}
''')
cvmat.add_registration_code('''
add_property( "data", bp::make_function(&CvMat_wrapper::get_data) )
''')


cc.write('''
#-----------------------------------------------------------------------------
# Multi-dimensional dense array (CvMatND)
#-----------------------------------------------------------------------------

CV_MATND_MAGIC_VAL    = 0x42430000
CV_TYPE_NAME_MATND    = "opencv-nd-matrix"

CV_MAX_DIM = 32
CV_MAX_DIM_HEAP = (1 << 16)


''')

# CvMatND
cvmatnd = mb.class_('CvMatND')
cvmatnd.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmatnd.var(z).exclude()
# deal with 'data'
cvmatnd.include_files.append( "boost/python/object.hpp" )
cvmatnd.include_files.append( "boost/python/str.hpp" )
cvmatnd.add_wrapper_code('''
static bp::object get_data( CvMatND const & inst ){        
    return bp::str((const char *)inst.data.ptr, inst.dim[0].step*inst.dim[0].size);
}
''')
cvmatnd.add_registration_code('''
add_property( "data", bp::make_function(&CvMatND_wrapper::get_data) )
''')

cc.write('''
#-----------------------------------------------------------------------------
# Multi-dimensional sparse array (CvSparseMat) 
#-----------------------------------------------------------------------------

CV_SPARSE_MAT_MAGIC_VAL    = 0x42440000
CV_TYPE_NAME_SPARSE_MAT    = "opencv-sparse-matrix"


''')

# CvSparseMat
cvsparsemat = mb.class_('CvSparseMat')
cvsparsemat.include()
for z in ('heap', 'hashtable'): # TODO: fix
    cvsparsemat.var(z).exclude()
    

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

# IplImage
iplimage = mb.class_('_IplImage')
iplimage.rename('IplImage')
iplimage.include()
for z in ('imageId', 'imageData', 'roi', 'imageDataOrigin', 'tileInfo', 'maskROI'): # don't need these attributes
    iplimage.var(z).exclude()
# deal with 'imageData'
iplimage.include_files.append( "boost/python/object.hpp" )
iplimage.include_files.append( "boost/python/str.hpp" )
iplimage.add_wrapper_code('''
static bp::object get_data( _IplImage const & inst ){        
    return bp::str(inst.imageData, inst.imageSize);
}
''')
iplimage.add_registration_code('''
add_property( "data", bp::make_function(&_IplImage_wrapper::get_data) )
''')

# CvArr
mb.class_('CvArr').include()

def add_underscore(decl):
    decl.rename('_'+decl.name)
    decl.include()


cc.write('''
#=============================================================================
# CxCore/Operations on Arrays
#=============================================================================


''')

cc.write('''
# IplImage
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
        z.add_transformation(input_double_pointee(0))
        
# cvInit... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvInit')):
    add_underscore(z)
    z.call_policies = CP.return_self()


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


# cvReleaseData
add_underscore(mb.free_fun('cvReleaseData'))

# -----------------------------------------------------------------------------------------------
# Final tasks
# -----------------------------------------------------------------------------------------------

for z in ('hdr_refcount', 'refcount'): # too low-level
    mb.decls(z).exclude() 

# mb.free_function( return_type='IplImage *' ).call_policies \
    # = call_policies.return_value_policy( call_policies.return_pointee_value )


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
    
# IplImage
# iplimage = mb.class_('_IplImage')
# for z in ('imageId', 'imageData', 'imageDataOrigin'):
    # iplimage.var(z).expose_address = True
    
# CvMat
# cvmat = mb.class_('CvMat')
# cvmat.include()
# cvmat.var('refcount').expose_address = True
# for z in ('ptr', 's', 'i', 'fl', 'db'):
    # cvmat.var(z).exclude()
# expose_addressof_member(cvmat, 'data')

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
