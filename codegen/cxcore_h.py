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
    for z in mb.free_funs(lambda decl: decl.name.startswith('cvRelease')):
        if not z.name in ('cvRelease', 'cvReleaseData'):
            FT.add_underscore(z)
            z._transformer_creators.append(FT.input_double_pointee(0))
            
    # cvCreateImageHeader
    FT.expose_func(mb.free_fun('cvCreateImageHeader'), ownershiplevel=1, return_pointee=True)

    # cvInitImageHeader
    FT.expose_func(mb.free_fun('cvInitImageHeader'), return_arg_index=1, return_pointee=True)

    # cvCreateImage
    FT.expose_func(mb.free_fun('cvCreateImage'), ownershiplevel=3, return_pointee=True)
            
    # cvCloneImage
    FT.expose_func(mb.free_fun('cvCloneImage'), ownershiplevel=3, return_pointee=True)

    # cv...ImageCOI functions
    mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageCOI' in decl.name).include()

    # cv...ImageROI functions
    mb.free_functions(lambda decl: decl.name.startswith('cv') and 'ImageROI' in decl.name).include()



    # cvCreateMatHeader
    z = mb.free_fun('cvCreateMatHeader')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
    cc.write('''
def cvCreateMatHeader(rows, cols, cvmat_type):
    """CvMat cvCreateMatHeader(int rows, int cols, int type)

    Creates new matrix header
    """
    z = _PE._cvCreateMatHeader(rows, cols, cvmat_type)
    if z is not None:
        z._owner = True
    return z

    ''')


    # cvInitMatHeader
    z = mb.free_fun('cvInitMatHeader')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(1, 5, CP.return_self())


    # cvCreateMat
    z = mb.free_fun('cvCreateMat')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
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

    # cvCloneMat
    z = mb.free_fun('cvCloneMat')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
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

    # cvScalarToRawData and cvRawDataToScalar # TODO: fix this

    # cvCreateMatNDHeader
    z = mb.free_fun('cvCreateMatNDHeader')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
    cc.write('''
def cvCreateMatNDHeader(sizes, type):
    """CvMatND cvCreateMatNDHeader(sequence_of_ints sizes, int type)

    Creates new matrix header
    """
    z = _PE._cvCreateMatNDHeader(sizes, type)
    if z is not None:
        z._owner = True
    return z

    ''')

    # cvCreateMatND
    z = mb.free_fun('cvCreateMatND')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
    cc.write('''
def cvCreateMatND(sizes, type):
    """CvMatND cvCreateMatND(sequence_of_ints sizes, int type)

    Creates multi-dimensional dense array
    """
    z = _PE._cvCreateMatND(sizes, type)
    if z is not None:
        z._owner = True
    return z

    ''')

    # cvInitMatNDHeader
    z = mb.free_fun('cvInitMatNDHeader')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(1, 4, CP.return_self())
    cc.write('''
def cvMatND(sizes, mattype, data=None):
    return cvInitMatNDHeader(CvMatND(), sizes, mattype, data=data)
    
    ''')

    # cvCloneMatND
    z = mb.free_fun('cvCloneMatND')
    FT.add_underscore(z)
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )
    cc.write('''
def cvCloneMatND(mat):
    """CvMatND cvCloneMatND(const CvMatND mat)

    Creates full copy of multi-dimensional array
    """
    z = _PE._cvCloneMatND(mat)
    if z is not None:
        z._owner = True
    return z

    ''')

    # cvCreateSparseMat
    z = mb.free_fun('cvCreateImageHeader')
    z.include()
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )

    # cvCloneSparseMat
    z = mb.free_fun('cvCloneSparseMat')
    z.include()
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )

    # cvInitSparseMatIterator
    z = mb.free_fun('cvInitSparseMatIterator')
    z.include()
    z.call_policies = CP.return_value_policy(CP.reference_existing_object, CP.with_custodian_and_ward_postcall(0, 2))

    # cvGetNextSparseNode
    z = mb.free_fun('cvGetNextSparseNode')
    z.include()
    z.call_policies = CP.return_value_policy(CP.reference_existing_object, CP.with_custodian_and_ward_postcall(0, 1))

    # CvNArrayIterator
    z = mb.class_('CvNArrayIterator')
    z.include()
    for t in ('ptr', 'hdr'): # TODO: fix this
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
    z._transformer_creators.append(FT.input_dynamic_array_of_pointers('arrs', 'count'))

    # functions
    for z in ('cvNextNArraySlice', 'cvGetElemType', 'cvGetDimSize'):
        mb.free_fun(z).include()

    # cvPtr*D # TODO: fix this

    # cvGet*D and cvSet*D
    mb.free_funs(lambda f: len(f.name) == 7 and f.name.startswith('cvGet') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 11 and f.name.startswith('cvGetReal') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 7 and f.name.startswith('cvSet') and f.name.endswith('D')).include()
    mb.free_funs(lambda f: len(f.name) == 11 and f.name.startswith('cvSetReal') and f.name.endswith('D')).include()
    for z in ('cvGetND', 'cvGetRealND', 'cvSetND', 'cvSetRealND', 'cvClearND'):
        mb.free_fun(z)._transformer_creators.append(FT.input_dynamic_array('idx'))

    # cvGetMat
    z = mb.free_fun('cvGetMat')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('coi'))
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))
    cc.write('''
def cvGetMat(arr, header=None, return_coi=False, allowND=0):
    """CvMat mat[, int output_coi] = cvGetMat(const CvArr arr, CvMat header=None, return_coi=None, int allowND=0)

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
    z = mb.free_fun('cvReshapeMatND')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(3, 1, CP.return_arg(3))
    z._transformer_creators.append(FT.input_dynamic_array('new_sizes', 'new_dims'))

    # cvReshape
    z = mb.free_fun('cvReshape')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(2, 1, CP.return_arg(2))

    # functions
    for z in ('cvRepeat', 'cvCreateData', 'cvReleaseData', 'cvSetData', 'cvGetSize'):
        mb.free_fun(z).include()
        
    # cvGetRawData # TODO: fix this

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
    z._transformer_creators.append(FT.input_dynamic_array_of_pointers('src', 'src_count'))
    z._transformer_creators.append(FT.input_dynamic_array_of_pointers('dst', 'dst_count'))
    z._transformer_creators.append(FT.input_dynamic_array('from_to', 'pair_count', remove_arg_size=False))
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
    z = mb.free_fun('cvRange')
    z.include()
    z.call_policies = CP.return_self()

    # cvCalcCovarMatrix
    z = mb.free_fun('cvCalcCovarMatrix')
    z.include()
    z._transformer_creators.append(FT.input_dynamic_array_of_pointers('vects', 'count'))    
        
        
    # Array Statistics
    cc.write('''
#-----------------------------------------------------------------------------
# Array Statistics
#-----------------------------------------------------------------------------

    
def cvMinMaxLoc(arr, min_loc=None, max_loc=None, mask=None):
    """double min_val, double max_val = cvMinMaxLoc(const CvArr arr, CvPoint min_loc=None, CvPoint max_loc=None, const CvArr mask=None)

    Finds global minimum and maximum in array or subarray, and optionally their locations
    [ctypes-opencv] If any of min_loc or max_loc is not None, it is filled with the resultant location.
    """
    min_val_p = _CT.c_double()
    max_val_p = _CT.c_double()
    
    _PE._cvMinMaxLoc(arr, min_val=_CT.addressof(min_val_p), max_val=_CT.addressof(max_val_p),
        min_loc=min_loc, max_loc=max_loc, mask=mask)
    
    return min_val_p.value, max_val_p.value
    

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
    z = mb.free_fun('cvCreateMemStorage')
    z.include()
    z.call_policies = CP.return_value_policy( CP.reference_existing_object )

    # cvCreateChildMemStorage
    z = mb.free_fun('cvCreateChildMemStorage')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 1, CP.return_value_policy(CP.reference_existing_object))
        
    # cvMemStorageAlloc, cvMemStorageAllocString # TODO: fix this

    # cvCreateSeq
    z = mb.free_fun('cvCreateSeq')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 4, CP.return_value_policy(CP.reference_existing_object))
        
    # cvSeq* # TODO: fix this

    # cvGetSeqElem, cvSeqElemIdx # TODO: fix

    # cvEndWriteSeq
    z = mb.free_fun('cvEndWriteSeq')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 1, CP.return_value_policy(CP.reference_existing_object))

    # cvCvtSeqToArray, cvMakeSeqHeaderForArray # TODO: fix

    # cvSeqSlice
    z = mb.free_fun('cvSeqSlice')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 3, CP.return_value_policy(CP.reference_existing_object))

    # cvCloneSeq
    z = mb.free_fun('cvCloneSeq')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 2, CP.return_value_policy(CP.reference_existing_object))

    # cvSeqSort, cvSeqSearch, cvSeqPartition # TODO: fix

    # cvChangeSeqBlock # TODO: fix

    # cvCreateSet
    z = mb.free_fun('cvCreateSet')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 4, CP.return_value_policy(CP.reference_existing_object))

    # cvSetAdd, cvSetNew, cvSetRemoveByPtr # TODO: fix

    # cvGetSetElem
    z = mb.free_fun('cvGetSetElem')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 1, CP.return_value_policy(CP.reference_existing_object))



    # CvGraph* and cvGraph* # TODO: fix this whole thing

    # CvTree* # TODO: fix this whole thing



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
    z._transformer_creators.append(FT.input_dynamic_double_array('pts', 'npts', 'contours'))

    # cvPolyLine
    z = mb.free_fun('cvPolyLine')
    z.include()
    z._transformer_creators.append(FT.input_dynamic_double_array('pts', 'npts', 'contours'))

    # CvFont
    z = mb.class_('CvFont')
    z.include()
    for t in ('ascii', 'greek', 'cyrillic'): # TODO: fix
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
    z = mb.free_fun('cvOpenFileStorage')
    z.include()
    z.call_policies = CP.with_custodian_and_ward_postcall(0, 2, CP.return_value_policy(CP.reference_existing_object))

    # cvWrite
    z = mb.free_fun('cvWrite')
    z.include()
    z._transformer_creators.append(FT.input_string('ptr'))

    for z in (
        'cvGetHashedKey', 'cvGetRootFileNode', 'cvGetFileNode', 'cvGetFileNodeByName',
        ):
        f = mb.free_fun(z)
        f.include()
        f.call_policies = CP.with_custodian_and_ward_postcall(0, 1, CP.return_value_policy(CP.reference_existing_object))

    for z in (
        'cvFirstType', 'cvFindType', # TODO: cvTypeOf
        ):
        f = mb.free_fun(z)
        f.include()
        f.call_policies = CP.return_value_policy(CP.reference_existing_object)


    # CvImage and CvMatrix are not necessary


