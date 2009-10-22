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
        'cvReleaseImageHeader', 'cvReleaseImage', 'cvReleaseMat', 'cvReleaseMatND',
        'cvReleaseSparseMat', 'cvReleaseMemStorage', 'cvReleaseFileStorage',
        ):
        f = mb.free_fun(z)
        FT.add_underscore(f)
        f._transformer_creators.append(FT.input_double_pointee(0))
            
    # cvCreateImageHeader
    FT.expose_func(mb.free_fun('cvCreateImageHeader'), ownershiplevel=1)

    # cvInitImageHeader
    FT.expose_func(mb.free_fun('cvInitImageHeader'), return_arg_index=1)

    # cvCreateImage
    FT.expose_func(mb.free_fun('cvCreateImage'), ownershiplevel=3)
            
    # cvCloneImage
    FT.expose_func(mb.free_fun('cvCloneImage'), ownershiplevel=3)

    # cv...ImageCOI functions
    mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageCOI' in decl.name).include()

    # cv...ImageROI functions
    mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageROI' in decl.name).include()

    # cvCreateMatHeader
    FT.expose_func(mb.free_fun('cvCreateMatHeader'), ownershiplevel=1)

    # cvInitMatHeader
    FT.expose_func(mb.free_fun('cvInitMatHeader'), ward_indices=(5,), return_arg_index=1)

    # cvCreateMat
    FT.expose_func(mb.free_fun('cvCreateMat'), ownershiplevel=1)

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

    # cvCloneMat
    FT.expose_func(mb.free_fun('cvCloneMat'), ownershiplevel=1)

    # cv...RefData
    mb.free_funs(lambda f: f.name.startswith('cv') and 'RefData' in f.name).include()

    # cvGetSubRect
    z = mb.free_fun('cvGetSubRect')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetSubRect(arr, submat, rect):
    """CvMat cvGetSubRect(const CvArr arr, CvMat submat, CvRect rect)

    Returns matrix header corresponding to the rectangular sub-array of input image or matrix
    [ctypes-opencv] If 'submat' is None, it is internally created.
    """
    if submat is None:
        submat = CvMat()
    _PE._cvGetSubRect(arr, submat, rect)
    return submat

cvGetSubArr = cvGetSubRect

    ''')

    # cvGetRows and cvGetRow
    z = mb.free_fun('cvGetRows')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetRows(arr, submat, start_row, end_row, delta_row=1):
    """CvMat cvGetRows(const CvArr arr, CvMat submat, int start_row, int end_row, int delta_row=1)

    Returns array row or row span
    [ctypes-opencv] If 'submat' is None, it is internally created.
    """
    if submat is None:
        submat = CvMat()
    _cvGetRows(arr, submat, start_row, end_row, delta_row=delta_row)
    return submat
    
def cvGetRow(arr, submat=None, row=0):
    return cvGetRows(arr, submat, row, row+1)

    ''')

    # cvGetCols and cvGetCol
    z = mb.free_fun('cvGetCols')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetCols(arr, submat, start_col, end_col):
    """CvMat cvGetCols(const CvArr arr, CvMat submat, int start_col, int end_col)

    Returns array column or column span
    [ctypes-opencv] If 'submat' is None, it is internally created.
    """
    if submat is None:
        submat = CvMat()
    _cvGetCols(arr, submat, start_col, end_col)
    return submat
    
def cvGetCol(arr, submat=None, col=0):
    return cvGetCols(arr, submat, col, col+1)

    ''')

    # cvGetDiag
    z = mb.free_fun('cvGetDiag')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetDiag(arr, submat=None, diag=0):
    """CvMat cvGetDiag(const CvArr arr, CvMat submat, int diag=0)

    Returns one of array diagonals
    [ctypes-opencv] If 'submat' is None, it is internally created.
    """
    if submat is None:
        submat = CvMat()
    _cvGetDiag(arr, submat, diag=diag)
    return submat

    ''')

    # cvScalarToRawData and cvRawDataToScalar # TODO: fix these funcs

    # cvCreateMatNDHeader
    FT.expose_func(mb.free_fun('cvCreateMatNDHeader'), ownershiplevel=1)

    # cvCreateMatND
    FT.expose_func(mb.free_fun('cvCreateMatND'), ownershiplevel=1)

    # cvInitMatNDHeader
    FT.expose_func(mb.free_fun('cvInitMatNDHeader'), ward_indices=(4,), return_arg_index=1)

    cc.write('''
def cvMatND(sizes, mattype, data=None):
    return cvInitMatNDHeader(CvMatND(), sizes, mattype, data=data)
    
    ''')

    # cvCloneMatND
    FT.expose_func(mb.free_fun('cvCloneMatND'), ownershiplevel=1)

    # cvCreateSparseMat
    FT.expose_func(mb.free_fun('cvCreateSparseMat'), ownershiplevel=1)

    # cvCloneSparseMat
    FT.expose_func(mb.free_fun('cvCloneSparseMat'), ownershiplevel=1)

    # cvInitSparseMatIterator
    FT.expose_func(mb.free_fun('cvInitSparseMatIterator'), ward_indices=(1, 2))

    # cvGetNextSparseNode
    FT.expose_func(mb.free_fun('cvGetNextSparseNode'), ward_indices=(1,))

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
    for z in ('cvNextNArraySlice', 'cvGetElemType', 'cvGetDimSize'):
        mb.free_fun(z).include()

    # cvGetDims()
    z = mb.free_fun('cvGetDims')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('sizes'))
    cc.write('''
def cvGetDims(arr, return_sizes=False):
    """
    z = cvGetDims((const CvArr)arr, (bool)return_sizes=False)

    Retrieves array dimensions.

    [pyopencv] If 'return_sizes' is True, z = a tuple containing the sizes of the dimensions.
    [pyopencv] If 'return_sizes' is False, z = an integer that is the number of dimensions.
    """
    if not return_sizes:
        return _PE._cvGetDims(arr, 0)

    sizes = (_CT.c_int*CV_MAX_DIM)()
    dims = _PE._cvGetDims(arr, _CT.addressof(sizes))
    return tuple(sizes[:dims])
    ''')

    # cvPtr*D # too low-level, wait until requested

    # cvGet*D and cvSet*D
    mb.free_funs(lambda f: len(f.name) == 7 and f.name.startswith('cvGet') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 11 and f.name.startswith('cvGetReal') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 7 and f.name.startswith('cvSet') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 11 and f.name.startswith('cvSetReal') and f.name.endswith('D')).include()
    for z in ('cvGetND', 'cvGetRealND', 'cvSetND', 'cvSetRealND', 'cvClearND'):
        mb.free_fun(z)._transformer_creators.append(FT.input_array1d('idx'))

    # cvGetMat
    z = mb.free_fun('cvGetMat')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('coi'))
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetMat(arr, header=None, return_coi=False, allowND=0):
    """CvMat mat[, int output_coi] = cvGetMat(const CvArr arr, CvMat header=None, return_coi=False, int allowND=0)

    Returns matrix header for arbitrary array
    [ctypes-opencv] If 'header' is None, it is internally created.
    [ctypes-opencv] If 'return_coi' is True, output_coi is returned.
    """
    if header is None:
        header = CvMat()
    if return_coi:
        coi = _CT.c_int()
        _PE._cvGetMat(arr, header, _CT.addressof(coi), allowND)
        return (header, coi.value)
        
    _PE._cvGetMat(arr, header, 0, allowND)
    return header
    
    ''')

    # cvGetImage
    z = mb.free_fun('cvGetImage')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetImage(arr, image_header=None):
    """IplImage cvGetImage(const CvArr arr, IplImage image_header=None)

    Returns image header for arbitrary array
    [ctypes-opencv] If 'image_header' is None, it is internally created.
    """
    if image_header is None:
        image_header = IplImage()
    _PE._cvGetImage(arr, image_header)
    return image_header

    ''')

    # cvReshapeMatND
    FT.expose_func(mb.free_fun('cvReshapeMatND'), ward_indices=(1,), return_arg_index=3, 
        transformer_creators=[FT.input_array1d('new_sizes', 'new_dims')])

    # cvReshape
    FT.expose_func(mb.free_fun('cvReshape'), ward_indices=(1,), return_arg_index=2) 

    # functions
    for z in ('cvRepeat', 'cvGetSize'):
        mb.free_fun(z).include()

    # cvCreateData
    z = mb.free_fun('cvCreateData')
    FT.add_underscore(z)
    cc.write('''
def cvCreateData(arr):
    cvReleaseData(arr) # release previous data first
    _PE._cvCreateData(arr)
    if isinstance(arr, IplImage):
        arr._ownershiplevel |= 2 # now arr owns data
cvCreateData.__doc__ = _PE._cvCreateData.__doc__
    ''')
    

    # cvReleaseData
    z = mb.free_fun('cvReleaseData')
    FT.add_underscore(z)
    z.include()
    cc.write('''
def cvReleaseData(arr):
    _PE._cvReleaseData(arr)
    arr._depends = None # remove previous links
    if isinstance(arr, IplImage):
        arr._ownershiplevel &= ~2 # arr does not own data anymore
cvReleaseData.__doc__ = _PE._cvReleaseData.__doc__

    ''')
        
    # cvSetData
    z = mb.free_fun('cvSetData')
    FT.add_underscore(z)
    z.call_policies = CP.with_custodian_and_ward(1, 2)
    z._transformer_creators.append(FT.input_string('data'))
    cc.write('''
def cvSetData(arr, data, step):
    cvReleaseData(arr)
    _PE._cvSetData(arr, data, step)
    arr._depends = (data,) # link to the current data
cvSetData.__doc__ = _PE._cvSetData.__doc__
    ''')
    mb.add_doc('cvSetData', 'data is a string')
        
    # cvGetRawData # too low-level, wait until requested

    # functions
    for z in ('cvCopy', 'cvSet', 'cvSetZero', 'cvSplit', 'cvMerge', 
        'cvConvertScale', 'cvConvertScaleAbs', 'cvCheckTermCriteria',
        ):
        mb.free_fun(z).include()
    cc.write('''
cvZero = cvSetZero

cvCvtScale = cvConvertScale

cvScale = cvConvertScale

def cvConvert(src, dst):
    cvConvertScale(src, dst, 1, 0)

cvCvtScaleAbs = cvConvertScaleAbs


    ''')
        
    # CvMixChannels
    z = mb.free_fun('cvMixChannels')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.input_array1d('src', 'src_count'))
    z._transformer_creators.append(FT.input_array1d('dst', 'dst_count'))
    z._transformer_creators.append(FT.input_array1d('from_to', 'pair_count', remove_arg_size=False))
    cc.write('''
def cvMixChannels(src, dst, from_to):
    """void cvMixChannels(sequence_of_CvArr src, sequence_of_CvArr dst, sequence_of_int from_to)
    
    Copies several channels from input arrays to certain channels of output arrays
    
    Example: cvMixChannels((z1, z2, z3), (t1, t2, t3), (0,1, 1,2, 2,0))
        where z1, z2, z3, t1, t2, t3 are instances of CvArr
    """
    return _PE._cvMixChannels(src, dst, from_to, len(from_to) >> 1)

    ''')
        
    # Arithmetic, logic and comparison operations
    cc.write('''
#-----------------------------------------------------------------------------
# Arithmetic, logic and comparison operations
#-----------------------------------------------------------------------------


def cvAXPY( A, real_scalar, B, C ):
    cvScaleAdd(A, cvRealScalar(real_scalar), B, C)
    
CV_CMP_EQ = 0
CV_CMP_GT = 1
CV_CMP_GE = 2
CV_CMP_LT = 3
CV_CMP_LE = 4
CV_CMP_NE = 5

def cvAbs(src, dst):
    """void cvAbs(const CvArr src, CvArr dst)
    
    Calculates absolute value of every element in array
    """
    cvAbsDiffS(src, dst, cvScalarAll(0))

    
    ''')
        
    # functions
    for z in (
        'cvAdd', 'cvAddS', 'cvSub', 'cvSubS', 'cvSubRS', 'cvMul', 'cvDiv', 
        'cvScaleAdd', 'cvAddWeighted', 'cvDotProduct', 'cvAnd', 'cvAndS', 
        'cvOr', 'cvOrS', 'cvXor', 'cvXorS', 'cvNot', 'cvInRange', 'cvInRangeS',
        'cvCmp', 'cvCmpS', 'cvMin', 'cvMax', 'cvMinS', 'cvMaxS', 'cvAbsDiff', 'cvAbsDiffS', 
        ):
        mb.free_fun(z).include()



    # Math operations
    cc.write('''
#-----------------------------------------------------------------------------
# Math operations
#-----------------------------------------------------------------------------


CV_CHECK_RANGE = 1
CV_CHECK_QUIET = 2

cvCheckArray = cvCheckArr

CV_RAND_UNI = 0
CV_RAND_NORMAL = 1

CV_SORT_EVERY_ROW = 0
CV_SORT_EVERY_COLUMN = 1
CV_SORT_ASCENDING = 0
CV_SORT_DESCENDING = 16

    
    ''')
        
    # functions
    for z in (
        'cvCartToPolar', 'cvPolarToCart', 'cvPow', 'cvExp', 'cvLog', 'cvFastArctan', 'cvCbrt', 
        'cvCheckArr', 'cvRandArr', 'cvRandShuffle', 'cvSort', 'cvSolveCubic', 'cvSolvePoly',
        ):
        mb.free_fun(z).include()


    # Matrix operations
    cc.write('''
#-----------------------------------------------------------------------------
# Matrix operations
#-----------------------------------------------------------------------------


CV_GEMM_A_T = 1
CV_GEMM_B_T = 2
CV_GEMM_C_T = 4

cvMatMulAddEx = cvGEMM

def cvMatMulAdd(src1, src2, src3, dst):
    """void cvMatMulAdd(const CvArr src1, const CvArr src2, const CvArr src3, CvArr dst)
    
    Performs dst = src1*src2+src3
    """
    cvGEMM(src1, src2, 1, src3, 1, dst, 0)

def cvMatMul(src1, src2, dst):
    """void cvMatMul(const CvArr src1, const CvArr src2, CvArr dst)
    
    Performs dst = src1*src2
    """
    cvMatMulAdd(src1, src2, 0, dst)

cvMatMulAddS = cvTransform

cvT = cvTranspose

cvMirror = cvFlip

CV_SVD_MODIFY_A = 1
CV_SVD_U_T = 2
CV_SVD_V_T = 4

CV_LU = 0
CV_SVD = 1
CV_SVD_SYM = 2
CV_CHOLESKY = 3
CV_QR = 4
CV_NORMAL = 16

cvInv = cvInvert

CV_COVAR_SCRAMBLED = 0
CV_COVAR_NORMAL = 1
CV_COVAR_USE_AVG = 2
CV_COVAR_SCALE = 4
CV_COVAR_ROWS = 8
CV_COVAR_COLS = 16

CV_PCA_DATA_AS_ROW = 0
CV_PCA_DATA_AS_COL = 1
CV_PCA_USE_AVG = 2

cvMahalonobis = cvMahalanobis

    
    ''')
        
    # functions
    for z in (
        'cvCrossProduct', 'cvGEMM', 'cvTransform', 'cvPerspectiveTransform', 'cvMulTransposed',
        'cvTranspose', 'cvCompleteSymm', 'cvFlip', 'cvSVD', 'cvSVBkSb', 
        'cvInvert', 'cvSolve', 'cvDet', 'cvTrace', 'cvEigenVV', 'cvSetIdentity',
        'cvCalcPCA', 'cvProjectPCA', 'cvBackProjectPCA', 'cvMahalanobis',
        ):
        mb.free_fun(z).include()

    # cvRange
    FT.expose_func(mb.free_fun('cvRange'), return_arg_index=1) 

    # cvCalcCovarMatrix
    z = mb.free_fun('cvCalcCovarMatrix')
    z.include()
    z._transformer_creators.append(FT.input_array1d('vects', 'count'))    
        
        
    # Array Statistics
    cc.write('''
#-----------------------------------------------------------------------------
# Array Statistics
#-----------------------------------------------------------------------------

    
CV_C = 1
CV_L1 = 2
CV_L2 = 4
CV_NORM_MASK = 7
CV_RELATIVE = 8
CV_DIFF = 16
CV_MINMAX = 32
CV_DIFF_C = (CV_DIFF | CV_C)
CV_DIFF_L1 = (CV_DIFF | CV_L1)
CV_DIFF_L2 = (CV_DIFF | CV_L2)
CV_RELATIVE_C = (CV_RELATIVE | CV_C)
CV_RELATIVE_L1 = (CV_RELATIVE | CV_L1)
CV_RELATIVE_L2 = (CV_RELATIVE | CV_L2)

CV_REDUCE_SUM = 0
CV_REDUCE_AVG = 1
CV_REDUCE_MAX = 2
CV_REDUCE_MIN = 3


    ''')

    # functions
    for z in (
        'cvSum', 'cvCountNonZero', 'cvAvg', 'cvAvgSdv',
        'cvNorm', 'cvNormalize', 'cvReduce',
        ):
        mb.free_fun(z).include()

    # cvMinMaxLoc
    z = mb.free_fun('cvMinMaxLoc')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('min_val'))
    z._transformer_creators.append(FT.from_address('max_val'))
    cc.write('''
def cvMinMaxLoc(arr, return_min_loc=False, return_max_loc=False, mask=None):
    """(double)min_val, (double)max_val[, (CvPoint)min_loc][, (CvPoint)max_loc] = cvMinMaxLoc((const CvArr)arr, (bool)return_min_loc=False, (bool)return_max_loc=False, const CvArr mask=None)

    Finds global minimum and maximum in array or subarray, and optionally their locations
    [pyopencv] 'min_loc' is returned if 'return_min_loc' is True. 
    [pyopencv] 'max_loc' is returned if 'return_max_loc' is True. 
    """
    min_val_p = _CT.c_double()
    max_val_p = _CT.c_double()

    min_loc = CvPoint() if return_min_loc is None else None
    max_loc = CvPoint() if return_max_loc is None else None
    
    _PE._cvMinMaxLoc(arr, min_val=_CT.addressof(min_val_p), max_val=_CT.addressof(max_val_p),
        min_loc=min_loc, max_loc=max_loc, mask=mask)
    
    z = (min_val_p.value, max_val_p.value)
    if min_loc is not None:
        z.append(min_loc)
    if max_loc is not None:
        z.append(max_loc)

    return z
    
    ''')



    # Discrete Linear Transforms and Related Functions
    cc.write('''
#-----------------------------------------------------------------------------
# Discrete Linear Transforms and Related Functions
#-----------------------------------------------------------------------------

    
# Performs forward or inverse Discrete Fourier transform of 1D or 2D floating-point array
CV_DXT_FORWARD = 0
CV_DXT_INVERSE = 1
CV_DXT_SCALE = 2     # divide result by size of array
CV_DXT_INV_SCALE = CV_DXT_SCALE | CV_DXT_INVERSE
CV_DXT_INVERSE_SCALE = CV_DXT_INV_SCALE
CV_DXT_ROWS = 4     # transfor each row individually
CV_DXT_MUL_CONJ = 8     # conjugate the second argument of cvMulSpectrums

cvFFT = cvDFT


    ''')

    # functions
    for z in (
        'cvDFT', 'cvMulSpectrums', 'cvGetOptimalDFTSize', 'cvDCT',
        ):
        mb.free_fun(z).include()


    # Dynamic Data Structure
    cc.write('''
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

cvDrawRect = cvRectangle
cvDrawLine = cvLine
cvDrawCircle = cvCircle
cvDrawEllipse = cvEllipse
cvDrawPolyLine = cvPolyLine

CV_FONT_HERSHEY_SIMPLEX = 0
CV_FONT_HERSHEY_PLAIN = 1
CV_FONT_HERSHEY_DUPLEX = 2
CV_FONT_HERSHEY_COMPLEX = 3
CV_FONT_HERSHEY_TRIPLEX = 4
CV_FONT_HERSHEY_COMPLEX_SMALL = 5
CV_FONT_HERSHEY_SCRIPT_SIMPLEX = 6
CV_FONT_HERSHEY_SCRIPT_COMPLEX = 7
CV_FONT_ITALIC = 16
CV_FONT_VECTOR0 = CV_FONT_HERSHEY_SIMPLEX


    ''')

    # functions
    for z in (
        'cvLine', 'cvRectangle', 'cvCircle', 'cvEllipse', 'cvEllipseBox', 'cvFillConvexPoly',
        'cvClipLine', 'cvInitLineIterator',
        'cvInitFont', 'cvFont', 'cvPutText',
        'cvColorToScalar',
        'cvDrawContours', 'cvLUT',
        ):
        mb.free_fun(z).include()

    # cvFillPoly
    z = mb.free_fun('cvFillPoly')
    z.include()
    z._transformer_creators.append(FT.input_array2d('pts', 'npts', 'contours'))

    # cvPolyLine
    z = mb.free_fun('cvPolyLine')
    z.include()
    z._transformer_creators.append(FT.input_array2d('pts', 'npts', 'contours'))

    # CvFont
    z = mb.class_('CvFont')
    z.include()
    for t in ('ascii', 'greek', 'cyrillic'): # wait until requested 
        z.var(t).exclude()

    # cvGetTextSize
    z = mb.free_fun('cvGetTextSize')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('baseline'))

    def cvGetTextSize(text_string, font):
        """(CvSize text_size, int baseline) = cvGetTextSize(string text_string, const CvFont font)

        Retrieves width and height of text string
        """
        text_size = CvSize()
        baseline = _CT.c_int()
        _PE.cvGetTextSize(text_string, font, text_size, _CT.addressof(baseline))
        return (text_size, baseline.value)


        
    # cvEllipse2Poly # TODO: fix


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
        'cvFirstType', 'cvFindType', # TODO: cvTypeOf
        ):
        FT.expose_func(mb.free_fun(z)) 


    # CvImage
    z = mb.class_('CvImage')
    mb.init_class(z)
    # deal with operators to convert into IplImage
    z.operators().exclude()
    z.include_files.append( "boost/python/object.hpp" )
    z.add_wrapper_code('''
static bp::object get_IplImage( CvImage const & inst ){
    return bp::object((const IplImage *)inst);
}
    ''')
    z.add_registration_code('''
add_property( "image", bp::make_function(&CvImage_wrapper::get_IplImage) )
    ''')
    z.mem_funs(lambda decl: decl.name == 'data').exclude() # wait until requested
    z.mem_funs(lambda decl: decl.name == 'roi_row').exclude() # wait until requested
    mb.finalize_class(z)

    # CvMatrix
    z = mb.class_('CvMatrix')
    mb.init_class(z)
    # deal with operators to convert into CvMat
    z.operators().exclude()
    z.include_files.append( "boost/python/object.hpp" )
    z.add_wrapper_code('''
static bp::object get_CvMat( CvMatrix const & inst ){
    return bp::object((const CvMat *)inst);
}
    ''')
    z.add_registration_code('''
add_property( "matrix", bp::make_function(&CvMatrix_wrapper::get_CvMat) )
    ''')
    z.mem_fun('set_data')._transformer_creators.append(FT.input_string('data'))
    z.mem_funs(lambda decl: decl.name == 'data').exclude() # wait until requested
    z.mem_funs(lambda decl: decl.name == 'row').exclude() # wait until requested
    mb.finalize_class(z)

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

