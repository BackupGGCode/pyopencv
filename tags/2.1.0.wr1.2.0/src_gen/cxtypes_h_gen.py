#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

import function_transformers as FT
import memvar_transformers as MT
import sdpypp
sb = sdpypp.SdModuleBuilder('cxtypes_h', number_of_files=3)


# basic data types
sb.register_decl('None', 'void')
sb.register_decl('bool', 'bool')
sb.register_decl('int8', 'char')
sb.register_decl('int8', 'signed char')
sb.register_decl('int8', 'schar')
sb.register_decl('uint8', 'unsigned char')
sb.register_decl('uint8', 'uchar')
sb.register_decl('int16', 'short')
sb.register_decl('int16', 'short int')
sb.register_decl('uint16', 'unsigned short')
sb.register_decl('uint16', 'short unsigned int')
sb.register_decl('uint16', 'ushort')
sb.register_decl('int', 'int')
sb.register_decl('uint', 'unsigned int')
sb.register_decl('long', 'long')
sb.register_decl('ulong', 'unsigned long')
sb.register_decl('int64', 'long long')
sb.register_decl('uint64', 'unsigned long long')
sb.register_decl('float32', 'float')
sb.register_decl('float64', 'double')



# register std::vector<std::string>
# sb.expose_class_vector('std::string')
# sb.register_decl('str', 'std::string')

# register vectors of basic data types
sb.expose_class_vector('char')
sb.expose_class_vector('unsigned char')
sb.expose_class_vector('short')
sb.expose_class_vector('unsigned short')
sb.expose_class_vector('int')
sb.expose_class_vector('unsigned int')
sb.expose_class_vector('long')
sb.expose_class_vector('unsigned long')
sb.expose_class_vector('long long')
sb.expose_class_vector('unsigned long long')
sb.expose_class_vector('float')
sb.expose_class_vector('double')
sb.expose_class_vector('unsigned char *', 'vector_string')
sb.expose_class_vector('std::vector< int >')
sb.expose_class_vector('std::vector< float >')

z = sb.dummy_struct
z.include_files.append("opencv_converters.hpp")
z.include_files.append("sequence.hpp")
sb.add_reg_code("sdcpp::register_sdobject<sdcpp::sequence>();")
sb.add_reg_code("sdcpp::register_sdobject<sdcpp::ndarray>();")


sb.register_ti('cv::Scalar_', ['double'], 'Scalar')

# expose some enumerations
sb.mb.enums(lambda x: x.name.startswith("Cv") or x.parent.name=="cv").include()    

sb.cc.write('''
import math as _m
import ctypes as _CT

#=============================================================================
# cxtypes.h
#=============================================================================


''')

# CvArr
# sb.mb.class_('CvArr').include()

sb.cc.write('''
#-----------------------------------------------------------------------------
# Common macros and inline functions
#-----------------------------------------------------------------------------

CV_PI = _m.pi
CV_LOG2 = 0.69314718055994530941723212145818

''')

# functions
for z in ('cvRound', 'cvFloor', 'cvCeil', 'cvIsNaN', 'cvIsInf'):
    sb.mb.free_fun(z).include()

sb.cc.write('''
#-----------------------------------------------------------------------------
# Random number generation
#-----------------------------------------------------------------------------

    
''')

sb.mb.add_declaration_code('''
struct CvRNG_to_python
{
    static PyObject* convert(CvRNG const& x)
    {
        return bp::incref(bp::object(cv::RNG(x)).ptr());
    }
};

''')
sb.mb.add_registration_code('bp::to_python_converter<CvRNG, CvRNG_to_python, false>();')


# CvMat
sb.cc.write('''
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

# CvMat -- for backward compatibility, mainly for ml.h
z = sb.mb.class_('CvMat')
sb.init_class(z)
for t in z.classes(''):
    try:
        t2 = t.var('ptr')
        t.exclude()
    except:
        pass
for t in ('refcount', 'hdr_refcount'):
    z.var(t).exclude()
z.add_declaration_code('''
static bp::object get_data(CvMat const &inst)
{
    return bp::object(bp::handle<>(PyBuffer_FromReadWriteMemory(
        (void*)inst.data.ptr, inst.rows*inst.step)));
}

''')
z.add_registration_code('add_property("data", &::get_data)')
sb.finalize_class(z)


# CvMatND
sb.cc.write('''
#-----------------------------------------------------------------------------
# Multi-dimensional dense array (CvMatND)
#-----------------------------------------------------------------------------

CV_MATND_MAGIC_VAL    = 0x42430000
CV_TYPE_NAME_MATND    = "opencv-nd-matrix"

CV_MAX_DIM = 32
CV_MAX_DIM_HEAP = (1 << 16)


#-----------------------------------------------------------------------------
# Multi-dimensional sparse array (CvSparseMat) 
#-----------------------------------------------------------------------------

CV_SPARSE_MAT_MAGIC_VAL    = 0x42440000
CV_TYPE_NAME_SPARSE_MAT    = "opencv-sparse-matrix"


''')

# CvHistogram -- disabled by Minh-Tri Pham
# cvhistogram = sb.mb.class_('CvHistogram')
# cvhistogram.include()
# for z in ('bins', 'thresh', 'thresh2'): # wait until requested
    # cvhistogram.var(z).exclude()
# sb.mb.decl('CvHistType').include()
# sb.cc.write('''
#-----------------------------------------------------------------------------
# Histogram
#-----------------------------------------------------------------------------

# CV_HIST_MAGIC_VAL     = 0x42450000
# CV_HIST_UNIFORM_FLAG  = (1 << 10)

# CV_HIST_RANGES_FLAG   = (1 << 11)

# CV_HIST_ARRAY         = 0
# CV_HIST_SPARSE        = 1
# CV_HIST_TREE          = CV_HIST_SPARSE

# CV_HIST_UNIFORM       = 1


# ''')
# sb.mb.insert_del_interface('CvHistogram', '_PE._cvReleaseHist')


# Other supplementary data type definitions
sb.cc.write('''
#-----------------------------------------------------------------------------
# Other supplementary data type definitions
#-----------------------------------------------------------------------------

# CV_TERMCRIT_ITER    = 1
# CV_TERMCRIT_NUMBER  = CV_TERMCRIT_ITER
# CV_TERMCRIT_EPS     = 2

''')

# CvRect -- now represented by cv::Rect
# z = sb.mb.class_('CvRect')
# sb.init_class(z)
# sb.finalize_class(z)
# sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + \\
        # ", width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
# KLASS.__repr__ = _KLASS__repr__
        
    # '''.replace("KLASS", z.alias))

    # CvSize -- now represented by cv::Size2i
    # CvSize2D32f -- now represented by cv::Size2f
    # for t in ('CvSize', 'CvSize2D32f'):
        # z = sb.mb.class_(t)
        # sb.init_class(z)
        # sb.finalize_class(z)
        # sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
# KLASS.__repr__ = _KLASS__repr__
        
        # '''.replace("KLASS", z.alias))

    # CvScalar -- use cv::Scalar instead
    # z = sb.mb.class_('CvScalar')
    # sb.init_class(z)
    # sb.finalize_class(z)
    # sb.finalize_class(z)
    # sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(" + self.ndarray.__str__() + ")"
# KLASS.__repr__ = _KLASS__repr__
        
# '''.replace("KLASS", z.alias))

# CvPoint, CvPoint2D32f, CvPoint2D64f  -- for backward compatibility
for t in ('CvPoint', 'CvPoint2D32f', 'CvPoint2D64f'):
    z = sb.mb.class_(t)
    sb.init_class(z)
    sb.finalize_class(z)
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ")"
KLASS.__repr__ = _KLASS__repr__
        
'''.replace("KLASS", z.alias))

# CvPoint3D32f, CvPoint3D64f  -- for backward compatibility
for t in ('CvPoint3D32f', 'CvPoint3D64f'):
    z = sb.mb.class_(t)
    sb.init_class(z)
    sb.finalize_class(z)
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ", z=" + repr(self.z) + ")"
KLASS.__repr__ = _KLASS__repr__
        
'''.replace("KLASS", z.alias))

# CvBox2D -- now represented by cv::RotatedRect
# z = sb.mb.class_('CvBox2D')
# sb.init_class(z)
# sb.finalize_class(z)
# sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(center=" + repr(self.center) + ", size=" + repr(self.size) + \\
        # ", angle=" + repr(self.angle) + ")"
# KLASS.__repr__ = _KLASS__repr__
        
    # '''.replace("KLASS", z.alias))

    # CvTermCriteria -- now represented by cv::TermCriteria
    # z = sb.mb.class_('CvTermCriteria')
    # sb.init_class(z)
    # sb.finalize_class(z)
    # sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(type=" + repr(self.type) + ", max_iter=" + repr(self.max_iter) + \\
        # ", epsilon=" + repr(self.epsilon) + ")"
# KLASS.__repr__ = _KLASS__repr__
        
    # '''.replace("KLASS", z.alias))

    # CvSlice -- now represented by cv::Range
    # z = sb.mb.class_('CvSlice')
    # sb.init_class(z)
    # sb.finalize_class(z)
    # sb.cc.write('''
# def _KLASS__repr__(self):
    # return "KLASS(start=" + repr(self.start_index) + ", end=" + repr(self.end_index) + ")"
# KLASS.__repr__ = _KLASS__repr__
        
# '''.replace("KLASS", z.alias))


# Dynamic Data structures
sb.cc.write('''
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

CV_TYPE_NAME_GRAPH = "opencv-graph"


''')

# CvMemBlock
z = sb.mb.class_('CvMemBlock')
sb.init_class(z)
for t in ('prev', 'next'):
    FT.expose_member_as_pointee(z, t)
sb.finalize_class(z)

# CvMemStorage -- now managed by cv::MemStorage
# this class is enabled only to let cv::MemStorage function properly
z = sb.mb.class_('CvMemStorage')
sb.init_class(z)
for t in ('bottom', 'top'):
    FT.expose_member_as_pointee(z, t)
sb.finalize_class(z)

# cv::MemStorage
sb.register_ti('cv::Ptr', ['CvMemStorage'], 'MemStorage')
sb.expose_class_Ptr('CvMemStorage')


# CvMemStoragePos
z = sb.mb.class_('CvMemStoragePos')
sb.init_class(z)
FT.expose_member_as_pointee(z, 'top')
sb.finalize_class(z)

# CvSeqBlock
z = sb.mb.class_('CvSeqBlock')
sb.init_class(z)
for t in ('prev', 'next'):
    FT.expose_member_as_pointee(z, t)
FT.expose_member_as_str(z, 'data')
sb.finalize_class(z)

# CvSeq
z = sb.mb.class_('CvSeq')
sb.init_class(z)
MT.expose_CvSeq_members(z, FT)
sb.finalize_class(z)
        
# CvSetElem
z = sb.mb.class_('CvSetElem')
sb.init_class(z)
FT.expose_member_as_pointee(z, 'next_free')
sb.finalize_class(z)

# CvSet
z = sb.mb.class_('CvSet')
sb.init_class(z)
MT.expose_CvSet_members(z, FT)
sb.finalize_class(z)


# CvGraphEdge
z = sb.mb.class_('CvGraphEdge')
sb.init_class(z)
for t in ('next', 'vtx'):
    FT.expose_member_as_array_of_pointees(z, t, 2)
sb.finalize_class(z)
    
# CvGraphVtx    
z = sb.mb.class_('CvGraphVtx')
sb.init_class(z)
FT.expose_member_as_pointee(z, 'first')
sb.finalize_class(z)

# CvGraphVtx2D
z = sb.mb.class_('CvGraphVtx2D')
sb.init_class(z)
FT.expose_member_as_pointee(z, 'first')
FT.expose_member_as_pointee(z, 'ptr')
sb.finalize_class(z)

# CvGraph
z = sb.mb.class_('CvGraph')
sb.init_class(z)
MT.expose_CvGraph_members(z, FT)
sb.finalize_class(z)

# CvChain
z = sb.mb.class_('CvChain')
sb.init_class(z)
MT.expose_CvSeq_members(z, FT)
sb.finalize_class(z)

# CvContour
z = sb.mb.class_('CvContour')
sb.init_class(z)
MT.expose_CvSeq_members(z, FT)
sb.mb.decl('CvPoint2DSeq').include()
sb.finalize_class(z)


# Sequence types
sb.cc.write('''
#-----------------------------------------------------------------------------
# Sequence types
#-----------------------------------------------------------------------------

#Viji Periapoilan 5/21/2007(start)

CV_SEQ_MAGIC_VAL            = 0x42990000

#define CV_IS_SEQ(seq) \
#    ((seq) != NULL && (((CvSeq*)(seq))->flags & CV_MAGIC_MASK) == CV_SEQ_MAGIC_VAL)

CV_SET_MAGIC_VAL           = 0x42980000
#define CV_IS_SET(set) \
#    ((set) != NULL && (((CvSeq*)(set))->flags & CV_MAGIC_MASK) == CV_SET_MAGIC_VAL)

CV_SEQ_ELTYPE_BITS         = 9
CV_SEQ_ELTYPE_MASK         =  ((1 << CV_SEQ_ELTYPE_BITS) - 1)
CV_SEQ_ELTYPE_POINT        =  CV_32SC2  #/* (x,y) */
CV_SEQ_ELTYPE_CODE         = CV_8UC1   #/* freeman code: 0..7 */
CV_SEQ_ELTYPE_GENERIC      =  0
CV_SEQ_ELTYPE_PTR          =  CV_USRTYPE1
CV_SEQ_ELTYPE_PPOINT       =  CV_SEQ_ELTYPE_PTR  #/* &(x,y) */
CV_SEQ_ELTYPE_INDEX        =  CV_32SC1  #/* #(x,y) */
CV_SEQ_ELTYPE_GRAPH_EDGE   =  0  #/* &next_o, &next_d, &vtx_o, &vtx_d */
CV_SEQ_ELTYPE_GRAPH_VERTEX =  0  #/* first_edge, &(x,y) */
CV_SEQ_ELTYPE_TRIAN_ATR    =  0  #/* vertex of the binary tree   */
CV_SEQ_ELTYPE_CONNECTED_COMP= 0  #/* connected component  */
CV_SEQ_ELTYPE_POINT3D      =  CV_32FC3  #/* (x,y,z)  */

CV_SEQ_KIND_BITS           = 3
CV_SEQ_KIND_MASK           = (((1 << CV_SEQ_KIND_BITS) - 1)<<CV_SEQ_ELTYPE_BITS)


# types of sequences
CV_SEQ_KIND_GENERIC        = (0 << CV_SEQ_ELTYPE_BITS)
CV_SEQ_KIND_CURVE          = (1 << CV_SEQ_ELTYPE_BITS)
CV_SEQ_KIND_BIN_TREE       = (2 << CV_SEQ_ELTYPE_BITS)

#Viji Periapoilan 5/21/2007(end)

# types of sparse sequences (sets)
CV_SEQ_KIND_GRAPH       = (3 << CV_SEQ_ELTYPE_BITS)
CV_SEQ_KIND_SUBDIV2D    = (4 << CV_SEQ_ELTYPE_BITS)

CV_SEQ_FLAG_SHIFT       = (CV_SEQ_KIND_BITS + CV_SEQ_ELTYPE_BITS)

# flags for curves
CV_SEQ_FLAG_CLOSED     = (1 << CV_SEQ_FLAG_SHIFT)
CV_SEQ_FLAG_SIMPLE     = (2 << CV_SEQ_FLAG_SHIFT)
CV_SEQ_FLAG_CONVEX     = (4 << CV_SEQ_FLAG_SHIFT)
CV_SEQ_FLAG_HOLE       = (8 << CV_SEQ_FLAG_SHIFT)

# flags for graphs
CV_GRAPH_FLAG_ORIENTED = (1 << CV_SEQ_FLAG_SHIFT)

CV_GRAPH               = CV_SEQ_KIND_GRAPH
CV_ORIENTED_GRAPH      = (CV_SEQ_KIND_GRAPH|CV_GRAPH_FLAG_ORIENTED)

# point sets
CV_SEQ_POINT_SET       = (CV_SEQ_KIND_GENERIC| CV_SEQ_ELTYPE_POINT)
CV_SEQ_POINT3D_SET     = (CV_SEQ_KIND_GENERIC| CV_SEQ_ELTYPE_POINT3D)
CV_SEQ_POLYLINE        = (CV_SEQ_KIND_CURVE  | CV_SEQ_ELTYPE_POINT)
CV_SEQ_POLYGON         = (CV_SEQ_FLAG_CLOSED | CV_SEQ_POLYLINE )
CV_SEQ_CONTOUR         = CV_SEQ_POLYGON
CV_SEQ_SIMPLE_POLYGON  = (CV_SEQ_FLAG_SIMPLE | CV_SEQ_POLYGON  )

# chain-coded curves
CV_SEQ_CHAIN           = (CV_SEQ_KIND_CURVE  | CV_SEQ_ELTYPE_CODE)
CV_SEQ_CHAIN_CONTOUR   = (CV_SEQ_FLAG_CLOSED | CV_SEQ_CHAIN)

# binary tree for the contour
CV_SEQ_POLYGON_TREE    = (CV_SEQ_KIND_BIN_TREE  | CV_SEQ_ELTYPE_TRIAN_ATR)

# sequence of the connected components
CV_SEQ_CONNECTED_COMP  = (CV_SEQ_KIND_GENERIC  | CV_SEQ_ELTYPE_CONNECTED_COMP)

# sequence of the integer numbers
CV_SEQ_INDEX           = (CV_SEQ_KIND_GENERIC  | CV_SEQ_ELTYPE_INDEX)

# CV_SEQ_ELTYPE( seq )   = ((seq)->flags & CV_SEQ_ELTYPE_MASK)
# CV_SEQ_KIND( seq )     = ((seq)->flags & CV_SEQ_KIND_MASK )

def CV_GET_SEQ_ELEM(TYPE, seq, index):
    result = cvGetSeqElem(seq, index)
    return cast(result, POINTER(TYPE))


''')

# CvSeqWriter -- obsolete, use Seq<T> instead
# z = sb.mb.class_('CvSeqWriter')
# sb.init_class(z)
# for t in ('seq', 'block'):
    # FT.expose_member_as_pointee(z, t)
# for t in ('ptr', 'block_min', 'block_max'):
    # FT.expose_member_as_str(z, t)
# sb.finalize_class(z)

# CvSeqReader -- obsolete, use Seq<T> instead
# z = sb.mb.class_('CvSeqReader')
# sb.init_class(z)
# MT.expose_CvSeqReader_members(z, FT)
# sb.finalize_class(z)


# Data structures for persistence (a.k.a serialization) functionality
sb.cc.write('''
#-----------------------------------------------------------------------------
# Data structures for persistence (a.k.a serialization) functionality
#-----------------------------------------------------------------------------

CV_STORAGE_READ = 0
CV_STORAGE_WRITE = 1
CV_STORAGE_WRITE_TEXT = CV_STORAGE_WRITE
CV_STORAGE_WRITE_BINARY = CV_STORAGE_WRITE
CV_STORAGE_APPEND = 2

CV_NODE_NONE        = 0
CV_NODE_INT         = 1
CV_NODE_INTEGER     = CV_NODE_INT
CV_NODE_REAL        = 2
CV_NODE_FLOAT       = CV_NODE_REAL
CV_NODE_STR         = 3
CV_NODE_STRING      = CV_NODE_STR
CV_NODE_REF         = 4 # not used
CV_NODE_SEQ         = 5
CV_NODE_MAP         = 6
CV_NODE_TYPE_MASK   = 7

def CV_NODE_TYPE(flags):
    return flags & CV_NODE_TYPE_MASK

# file node flags
CV_NODE_FLOW        = 8 # used only for writing structures to YAML format
CV_NODE_USER        = 16
CV_NODE_EMPTY       = 32
CV_NODE_NAMED       = 64

def CV_NODE_IS_INT(flags):
    return CV_NODE_TYPE(flags) == CV_NODE_INT
    
def CV_NODE_IS_REAL(flags):
    return CV_NODE_TYPE(flags) == CV_NODE_REAL
    
def CV_NODE_IS_STRING(flags):
    return CV_NODE_TYPE(flags) == CV_NODE_STRING
    
def CV_NODE_IS_SEQ(flags):
    return CV_NODE_TYPE(flags) == CV_NODE_SEQ
    
def CV_NODE_IS_MAP(flags):
    return CV_NODE_TYPE(flags) == CV_NODE_MAP
    
def CV_NODE_IS_COLLECTION(flags):
    return CV_NODE_TYPE(flags) >= CV_NODE_SEQ
    
def CV_NODE_IS_FLOW(flags):
    return bool(flags & CV_NODE_FLOW)
    
def CV_NODE_IS_EMPTY(flags):
    return bool(flags & CV_NODE_EMPTY)
    
def CV_NODE_IS_USER(flags):
    return bool(flags & CV_NODE_USER)
    
def CV_NODE_HAS_NAME(flags):
    return bool(flags & CV_NODE_NAMED)

CV_NODE_SEQ_SIMPLE = 256
def CV_NODE_SEQ_IS_SIMPLE(seq):
    return bool(seq[0].flags & CV_NODE_SEQ_SIMPLE)


''')

# CvFileStorage
sb.register_ti('CvFileStorage')
# z = sb.mb.class_('CvFileStorage')
# z.include()
# sb.mb.insert_del_interface('CvFileStorage', '_PE._cvReleaseFileStorage')

# CvAttrList
z = sb.mb.class_('CvAttrList')
sb.init_class(z)
z.var('attr').exclude()
# deal with 'attr'
z.include_files.append( "boost/python/object.hpp" )
z.include_files.append( "boost/python/str.hpp" )
z.include_files.append( "boost/python/list.hpp" )
z.include_files.append( "boost/python/tuple.hpp" )
z.add_wrapper_code('''
static bp::object get_attr( CvString const & inst ){
    if(!inst.ptr) return bp::object();
    bp::list l;
    for(int i = 0; inst.ptr[i]; ++i) l.append(inst.ptr[i]);
    return bp::tuple(l);
}
''')
z.add_registration_code('''
add_property( "attr", bp::make_function(&CvAttrList_wrapper::get_attr) )
''')
sb.finalize_class(z)

# CvTypeInfo
z = sb.mb.class_('CvTypeInfo')
sb.init_class(z)
for t in ('prev', 'next'):
    FT.expose_member_as_pointee(z, t)
FT.expose_member_as_str(z, 'type_name')
sb.finalize_class(z)

# CvString
z = sb.mb.class_('CvString')
sb.init_class(z)
for t in ('len', 'ptr'):
    z.var(t).exclude()
# deal with 'data'
z.include_files.append( "boost/python/object.hpp" )
z.include_files.append( "boost/python/str.hpp" )
z.add_wrapper_code('''
static bp::object get_data( CvString const & inst ){        
    return inst.ptr? bp::str((const char *)inst.ptr, inst.len) : bp::object();
}
''')
z.add_registration_code('''
add_property( "data", bp::make_function(&CvString_wrapper::get_data) )
''')
sb.finalize_class(z)


# CvStringHashNode
z = sb.mb.class_('CvStringHashNode')
z.include()
FT.expose_member_as_pointee(z, 'next')

# CvGenericHash
z = sb.mb.class_('CvGenericHash')
sb.init_class(z)
sb.finalize_class(z)

# CvFileNode -- now managed by cv::FileNode
# z = sb.mb.class_('CvFileNode')
# z.include()
# FT.expose_member_as_pointee(z, 'info')
# deal with 'data' -- wait until requested
# z.var('data').expose_address = True
# for t in ('f', 'i', 'str', 'seq', 'map'):
    # z.var(t).exclude()
    
# CvPluginFuncInfo
z = sb.mb.class_('CvPluginFuncInfo')
sb.init_class(z)
for t in ('func_addr', 'default_func_addr'):
    z.var(t).expose_address = True
FT.expose_member_as_str(z, 'func_names')    
sb.finalize_class(z)
    
# CvModuleInfo    
z = sb.mb.class_('CvModuleInfo')
sb.init_class(z)
for t in ('next', 'func_tab'):
    FT.expose_member_as_pointee(z, t)
for t in ('name', 'version'):
    FT.expose_member_as_str(z, t)
sb.finalize_class(z)

    
sb.done()
sb.save_regs('cxtypes_h_reg.sdd')
