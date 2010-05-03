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

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cxcore.h
#=============================================================================


    ''')
       

    cc.write('''
#-----------------------------------------------------------------------------
# Array allocation, deallocation, initialization and access to elements
#-----------------------------------------------------------------------------


    ''')

    # cvRelease... functions
    for z in (
        'cvReleaseFileStorage',
        ):
        f = mb.free_fun(z)
        FT.add_underscore(f)
        f._transformer_creators.append(FT.input_double_pointee(0))
            
    # CvNArrayIterator
    z = mb.class_('CvNArrayIterator')
    z.include()
    for t in ('ptr', 'hdr'): # wait until requested 
        z.var(t).exclude()
    cc.write('''
CV_MAX_ARR = 10

CV_NO_DEPTH_CHECK     = 1
CV_NO_CN_CHECK        = 2
CV_NO_SIZE_CHECK      = 4

    ''')


    # cvInitNArrayIterator
    z = mb.free_fun('cvInitNArrayIterator')
    z.include()
    z._transformer_creators.append(FT.input_as_list_of_Mat('arrs', 'count'))
   

    # functions
    for z in ('cvNextNArraySlice', 'cvGetElemType'):
        mb.free_fun(z).include()

    # Arithmetic, logic and comparison operations
    cc.write('''
#-----------------------------------------------------------------------------
# Arithmetic, logic and comparison operations
#-----------------------------------------------------------------------------

    
    ''')
        

    # Math operations
    cc.write('''
#-----------------------------------------------------------------------------
# Math operations
#-----------------------------------------------------------------------------


CV_RAND_UNI = 0
CV_RAND_NORMAL = 1

    
    ''')
        
    # missing functions
    for z in ('cvRandArr', 'cvSolveCubic', 'cvSolvePoly'):
        mb.free_fun(z).include()


    # Matrix operations
    cc.write('''
#-----------------------------------------------------------------------------
# Matrix operations
#-----------------------------------------------------------------------------


CV_COVAR_SCRAMBLED = 0
CV_COVAR_NORMAL = 1
CV_COVAR_USE_AVG = 2
CV_COVAR_SCALE = 4
CV_COVAR_ROWS = 8
CV_COVAR_COLS = 16

CV_PCA_DATA_AS_ROW = 0
CV_PCA_DATA_AS_COL = 1
CV_PCA_USE_AVG = 2

    ''')
    
    # cvRange
    z = mb.free_fun('cvRange')
    FT.expose_func(z, return_arg_index=1)
    z.rename('range_') # to avoid conflict with Python's range builtin function

        
    # Array Statistics
    cc.write('''
#-----------------------------------------------------------------------------
# Array Statistics
#-----------------------------------------------------------------------------

    
CV_REDUCE_SUM = 0
CV_REDUCE_AVG = 1
CV_REDUCE_MAX = 2
CV_REDUCE_MIN = 3


#-----------------------------------------------------------------------------
# Discrete Linear Transforms and Related Functions
#-----------------------------------------------------------------------------

    

#-----------------------------------------------------------------------------
# Dynamic Data Structure
#-----------------------------------------------------------------------------

    
CV_FRONT = 1
CV_BACK = 0


    ''')

    # functions
    for z in (
        'cvSliceLength', 'cvClearMemStorage', 'cvSaveMemStoragePos', 'cvRestoreMemStoragePos',
        'cvSetSeqBlockSize', 
        # 'cvSeqRemove', 'cvClearSeq', # these functions are now obsolete, use Seq<T> instead
        # 'cvStartAppendToSeq', 'cvStartWriteSeq', 'cvFlushSeqWriter', # these functions are now obsolete, use Seq<T> instead
        # 'cvStartReadSeq', 'cvGetSeqReaderPos', 'cvSetSeqReaderPos', # these functions are now obsolete, use Seq<T> instead
        # 'cvSeqRemoveSlice', 'cvSeqInsertSlice', 'cvSeqInvert', # these functions are now obsolete, use Seq<T> instead
        # 'cvCreateSeqBlock', # these functions are now obsolete, use Seq<T> instead
        'cvSetRemove', 'cvClearSet',
        ):
        mb.free_fun(z).include()

    # cvCreateMemStorage -- use sdopencv::createMemStorage
    mb.free_fun('createMemStorage').include()

    # cvCreateChildMemStorage -- use sdopencv::createChildMemStorage
    FT.expose_func(mb.free_fun('createChildMemStorage'), return_pointee=False, ward_indices=(1,)) 
            
    # cvMemStorageAlloc -- too low-level, removed

    # cvMemStorageAllocString
    z = mb.free_fun('cvMemStorageAllocString')
    FT.expose_func(z, ward_indices=(1,), return_pointee=False)

    # cvCreateSeq
    FT.expose_func(mb.free_fun('cvCreateSeq'), ward_indices=(4,)) 
        
    # cvSeq* # these functions are now obsolete, use Seq<T> instead

    # cvGetSeqElem, cvSeqElemIdx # these functions are now obsolete, use Seq<T> instead

    # cvEndWriteSeq # these functions are now obsolete, use Seq<T> instead
    # FT.expose_func(mb.free_fun('cvEndWriteSeq'), ward_indices=(1,)) 

    # cvCvtSeqToArray, cvMakeSeqHeaderForArray # too low-level

    # cvSeqSlice # these functions are now obsolete, use Seq<T> instead
    # FT.expose_func(mb.free_fun('cvSeqSlice'), ward_indices=(3,)) 

    # cvCloneSeq
    FT.expose_func(mb.free_fun('cvCloneSeq'), ward_indices=(2,)) 

    # cvSeqSort, cvSeqSearch, cvSeqPartition # wait until requested: sorting requires CmpFunc

    # cvChangeSeqBlock # these functions are now obsolete, use Seq<T> instead

    # cvCreateSet
    FT.expose_func(mb.free_fun('cvCreateSet'), ward_indices=(4,)) 

    # cvSetAdd, cvSetNew, cvSetRemoveByPtr # TODO: fix

    # cvGetSetElem
    FT.expose_func(mb.free_fun('cvGetSetElem'), ward_indices=(1,)) 


    # CvGraphScanner
    z = mb.class_('CvGraphScanner')
    z.include()
    for t in ('vtx', 'dst', 'edge', 'graph', 'stack'):
        FT.expose_member_as_pointee(z, t)
    mb.insert_del_interface('CvGraphScanner', '_PE._cvReleaseGraphScanner')


    # this whole set of functions cvGraph*, # TODO: fix these funcs



    # CvTreeNodeIterator
    z = mb.class_('CvTreeNodeIterator')
    z.include()
    z.var('node').expose_address = True # wait until requested


    # this whole set of functions cvTree*



    # Drawing Functions
    cc.write('''
#-----------------------------------------------------------------------------
# Drawing Functions
#-----------------------------------------------------------------------------

    
CV_FILLED = -1
CV_AA = 16

# Constructs a color value
def CV_RGB(r, g, b):
    return Scalar(b, g, r)

    
    ''')


    # System Functions
    cc.write('''
#-----------------------------------------------------------------------------
# System Functions
#-----------------------------------------------------------------------------

    
# Sets the error mode
CV_ErrModeLeaf = 0
CV_ErrModeParent = 1
CV_ErrModeSilent = 2


    ''')

    # functions
    for z in (
        'cvRegisterModule', 'cvUseOptimized', 'cvGetErrStatus', 'cvSetErrStatus',
        'cvGetErrMode', 'cvSetErrMode', 'cvError',
        'cvErrorStr', 'cvErrorFromIppStatus',
        ):
        mb.free_fun(z).include()
        
    # TODO: fix these functions:
    # cvGetModuleInfo, cvGetErrInfo, cvRedirectError, cvNulDevReport, cvStdErrReport, cvGuiBoxReport
    # cvSetMemoryManager, cvSetIPLAllocators


    # Data Persistence
    cc.write('''
#-----------------------------------------------------------------------------
# Data Persistence
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# CPU capabilities
#-----------------------------------------------------------------------------

CV_CPU_NONE    = 0    
CV_CPU_MMX     = 1
CV_CPU_SSE     = 2
CV_CPU_SSE2    = 3
CV_CPU_SSE3    = 4
CV_CPU_SSSE3   = 5
CV_CPU_SSE4_1  = 6
CV_CPU_SSE4_2  = 7
CV_CPU_AVX    = 10
CV_HARDWARE_MAX_FEATURE = 255
    

    ''')

    # functions
    for z in (
        'cvAttrValue', 'cvStartWriteStruct', 'cvEndWriteStruct',
        'cvWriteComment', 'cvStartNextStream',
        'cvWriteFileNode', 
        'cvRegisterType', 'cvUnregisterType',
        'cvGetTickCount', 'cvGetTickFrequency',
        'cvGetNumThreads', 'cvSetNumThreads', 'cvGetThreadNum',
        ):
        mb.free_fun(z).include()
        
    # cvWrite
    z = mb.free_fun('cvWrite')
    z.include()
    z._transformer_creators.append(FT.input_string('ptr'))
    
    # cvGetHashedKey
    FT.expose_func(mb.free_fun('cvGetHashedKey'), ward_indices=(1,)) 

    for z in (
        'cvFirstType', 'cvFindType', 
        ):
        FT.expose_func(mb.free_fun(z)) 
    FT.expose_func(mb.free_fun('cvTypeOf'), transformer_creators=[FT.fix_type('struct_ptr', '::CvArr *')])

    # CvModule
    z = mb.class_('CvModule')
    z.include()
    for t in ('info', 'first', 'last'):
        FT.expose_member_as_pointee(z, t)

    # CvType
    z = mb.class_('CvType')
    z.include()
    for t in ('info', 'first', 'last'):
        FT.expose_member_as_pointee(z, t)

