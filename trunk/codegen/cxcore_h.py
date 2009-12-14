#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.0 using Boost.Python and NumPy

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
        'cvReleaseMemStorage', 'cvReleaseFileStorage',
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
    z._transformer_creators.append(FT.input_array1d('arrs', 'count'))
   

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
    z = mb.free_fun('cvRandArr')
    z.include()
    z.rename('randArr')
    
    z = mb.free_fun('cvSolveCubic')
    z.include()
    z.rename('solveCubic')
    
    z = mb.free_fun('cvSolvePoly')
    z.include()
    z.rename('solvePoly')


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
    FT.expose_func(mb.free_fun('cvRange'), return_arg_index=1)

        
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
        'cvSeqRemove', 'cvClearSeq',
        'cvStartAppendToSeq', 'cvStartWriteSeq', 'cvFlushSeqWriter',
        'cvStartReadSeq', 'cvGetSeqReaderPos', 'cvSetSeqReaderPos',
        'cvSeqRemoveSlice', 'cvSeqInsertSlice', 'cvSeqInvert',
        'cvCreateSeqBlock',
        'cvSetRemove', 'cvClearSet',
        ):
        mb.free_fun(z).include()

    # cvCreateMemStorage
    FT.expose_func(mb.free_fun('cvCreateMemStorage'), ownershiplevel=1) 

    # cvCreateChildMemStorage
    FT.expose_func(mb.free_fun('cvCreateChildMemStorage'), ward_indices=(1,)) 
        
    # cvMemStorageAlloc -- too low-level, wait until requested

    # cvMemStorageAllocString
    z = mb.free_fun('cvMemStorageAllocString')
    FT.expose_func(z, ward_indices=(1,), return_pointee=False)

    # cvCreateSeq
    FT.expose_func(mb.free_fun('cvCreateSeq'), ward_indices=(4,)) 
        
    # cvSeq* # TODO: fix these whole functions

    # cvGetSeqElem, cvSeqElemIdx # TODO: fix these functions

    # cvEndWriteSeq
    FT.expose_func(mb.free_fun('cvEndWriteSeq'), ward_indices=(1,)) 

    # cvCvtSeqToArray, cvMakeSeqHeaderForArray # TODO: fix these funcs

    # cvSeqSlice
    FT.expose_func(mb.free_fun('cvSeqSlice'), ward_indices=(3,)) 

    # cvCloneSeq
    FT.expose_func(mb.free_fun('cvCloneSeq'), ward_indices=(2,)) 

    # cvSeqSort, cvSeqSearch, cvSeqPartition # TODO: fix these funcs

    # cvChangeSeqBlock # TODO: fix this func

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
    return CvScalar(b, g, r)

    
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

    

    ''')

    # functions
    for z in (
        'cvAttrValue', 'cvStartWriteStruct', 'cvEndWriteStruct', 'cvWriteInt', 'cvWriteReal',
        'cvWriteString', 'cvWriteComment', 'cvStartNextStream',
        'cvReadInt', 'cvReadIntByName', 'cvReadReal', 'cvReadRealByName', 'cvReadString', 'cvReadStringByName',
        'cvStartReadRawData', 'cvWriteFileNode', 'cvGetFileNodeName',
        'cvRegisterType', 'cvUnregisterType',
        'cvGetTickCount', 'cvGetTickFrequency',
        'cvGetNumThreads', 'cvSetNumThreads', 'cvGetThreadNum',
        ):
        mb.free_fun(z).include()
        
    # TODO: fix these functions:
    # cvWriteRawData, cvRead, cvReadByName, cvReadRawDataSlice, cvReadRawData,
    # cvSave, cvLoad

    # cvOpenFileStorage
    FT.expose_func(mb.free_fun('cvOpenFileStorage'), ward_indices=(2,), ownershiplevel=1) 

    # cvWrite
    z = mb.free_fun('cvWrite')
    z.include()
    z._transformer_creators.append(FT.input_string('ptr'))

    for z in (
        'cvGetHashedKey', 'cvGetRootFileNode', 'cvGetFileNode', 'cvGetFileNodeByName',
        ):
        FT.expose_func(mb.free_fun(z), ward_indices=(1,)) 

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

