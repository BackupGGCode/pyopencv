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


def expose_CvSeq_members(z, FT):
    z.include()
    for t in ('h_prev', 'h_next', 'v_prev', 'v_next', 'storage', 'free_blocks', 'first'):
        FT.expose_member_as_pointee(z, t)
    for t in ('block_max', 'ptr'):
        FT.expose_member_as_str(z, t)

def expose_CvSet_members(z, FT):
    expose_CvSeq_members(z, FT)
    FT.expose_member_as_pointee(z, 'free_elems')

def expose_CvSeqReader_members(z, FT):
    for t in ('seq', 'block'):
        FT.expose_member_as_pointee(z, t)
    for t in ('ptr', 'block_min', 'block_max', 'prev_elem'):
        FT.expose_member_as_str(z, t)

def generate_code(mb, cc, D, FT, CP):
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
IplImage._ownershiplevel = 0 # default: owns nothing
        
def _IplImage__del__(self):
    if self._ownershiplevel == 1: # own header only
        _PE._cvReleaseImageHeader(self)
    elif self._ownershiplevel == 2: # own data but not header
        _PE._cvReleaseData(self)
    elif self._ownershiplevel == 3: # own header and data
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
        _PE._cvReleaseMatND(self)
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
        'CvTermCriteria', 'cvTermCriteria', 
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

CV_TYPE_NAME_GRAPH = "opencv-graph"


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
    cc.write('''
def _CvMemStorage__del__(self):
    _PE._cvReleaseMemStorage(self)
CvMemStorage.__del__ = _CvMemStorage__del__

    ''')

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
    expose_CvSeq_members(z, FT)
            
    # CvSetElem
    z = mb.class_('CvSetElem')
    z.include()
    FT.expose_member_as_pointee(z, 'next_free')

    # CvSet
    z = mb.class_('CvSet')
    expose_CvSet_members(z, FT)


    # CvGraphEdge
    z = mb.class_('CvGraphEdge')
    z.include()
    for t in ('next', 'vtx'):
        FT.expose_member_as_array_of_pointees(z, t, 2)
        
    # CvGraphVtx    
    z = mb.class_('CvGraphVtx')
    z.include()
    FT.expose_member_as_pointee(z, 'first')

    # CvGraphVtx2D
    z = mb.class_('CvGraphVtx2D')
    z.include()
    FT.expose_member_as_pointee(z, 'first')
    FT.expose_member_as_pointee(z, 'ptr')

    # CvGraph
    z = mb.class_('CvGraph')
    expose_CvSet_members(z, FT)
    FT.expose_member_as_pointee(z, 'edges')

    # CvChain
    z = mb.class_('CvChain')
    expose_CvSeq_members(z, FT)

    # CvContour
    z = mb.class_('CvContour')
    expose_CvSeq_members(z, FT)
    mb.decl('CvPoint2DSeq').include()


    # Sequence types
    cc.write('''
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

    # CvSeqWriter
    z = mb.class_('CvSeqWriter')
    z.include()
    for t in ('seq', 'block'):
        FT.expose_member_as_pointee(z, t)
    for t in ('ptr', 'block_min', 'block_max'):
        FT.expose_member_as_str(z, t)

    # CvSeqReader
    z = mb.class_('CvSeqReader')
    z.include()
    expose_CvSeqReader_members(z, FT)


    # Data structures for persistence (a.k.a serialization) functionality
    cc.write('''
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
    z = mb.class_('CvFileStorage')
    z.include()
    cc.write('''
def _CvFileStorage__del__(self):
    _PE._cvReleaseFileStorage(self)
CvFileStorage.__del__ = _CvFileStorage__del__

    ''')

    # CvAttrList  # TODO: fix this
    z = mb.class_('CvAttrList')
    z.include()
    z.var('attr').exclude()

    # CvTypeInfo
    z = mb.class_('CvTypeInfo')
    z.include()
    for t in ('prev', 'next'):
        FT.expose_member_as_pointee(z, t)
    FT.expose_member_as_str(z, 'type_name')

    # CvString
    z = mb.class_('CvString')
    z.include()
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


    # CvStringHashNode
    z = mb.class_('CvStringHashNode')
    z.include()
    FT.expose_member_as_pointee(z, 'next')

    # CvGenericHash # TODO: fix this

    # CvFileNode
    z = mb.class_('CvFileNode')
    z.include()
    FT.expose_member_as_pointee(z, 'info')
    for t in ('data', 'f', 'i', 'str', 'seq', 'map'):
        z.var(t).exclude() # TODO: fix this

        
    # CvPluginFuncInfo
    z = mb.class_('CvPluginFuncInfo')
    z.include()
    for t in ('func_addr', 'default_func_addr'): # TODO: fix this
        z.var(t).exclude()
    FT.expose_member_as_str(z, 'func_names')    
        
    # CvModuleInfo    
    z = mb.class_('CvModuleInfo')
    z.include()
    for t in ('next', 'func_tab'):
        FT.expose_member_as_pointee(z, t)
    for t in ('name', 'version'):
        FT.expose_member_as_str(z, t)

    
    
