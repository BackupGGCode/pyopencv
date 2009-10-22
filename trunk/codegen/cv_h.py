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
# cv.h
#=============================================================================


    ''')

    # Data Structures in cv.h
    # pointers which are not Cv... * are excluded until further requested
    for z in (
        'CvFeatureTree', 'CvLSH', 'CvLSHOperations',
        'CvSURFPoint', 'CvSURFParams',
        'CvMSERParams', 
        'CvStarKeypoint',
        'CvStarDetectorParams', 
        'CvPOSITObject',
        'CvStereoBMState', 'CvStereoGCState', 
        ):
        k = mb.class_(z)
        k.include()
        try:
            vv = k.vars()
        except RuntimeError:
            vv = []
        for v in vv:
            if D.is_pointer(v.type):
                if 'Cv' in v.type.decl_string:
                    FT.expose_member_as_pointee(k, v.name)
                else:
                    v.exclude()


    # Image Processing
    cc.write('''
#-----------------------------------------------------------------------------
# Image Processing
#-----------------------------------------------------------------------------

    
CV_BLUR_NO_SCALE = 0
CV_BLUR = 1
CV_GAUSSIAN = 2
CV_MEDIAN = 3
CV_BILATERAL = 4

CV_INPAINT_NS = 0
CV_INPAINT_TELEA = 1

CV_SCHARR = -1
CV_MAX_SOBEL_KSIZE = 7

CV_BGR2BGRA =   0
CV_RGB2RGBA =   CV_BGR2BGRA

CV_BGRA2BGR =   1
CV_RGBA2RGB =   CV_BGRA2BGR

CV_BGR2RGBA =   2
CV_RGB2BGRA =   CV_BGR2RGBA

CV_RGBA2BGR =   3
CV_BGRA2RGB =   CV_RGBA2BGR

CV_BGR2RGB  =   4
CV_RGB2BGR  =   CV_BGR2RGB

CV_BGRA2RGBA =  5
CV_RGBA2BGRA =  CV_BGRA2RGBA

CV_BGR2GRAY =   6
CV_RGB2GRAY =   7
CV_GRAY2BGR =   8
CV_GRAY2RGB =   CV_GRAY2BGR
CV_GRAY2BGRA =  9
CV_GRAY2RGBA =  CV_GRAY2BGRA
CV_BGRA2GRAY =  10
CV_RGBA2GRAY =  11

CV_BGR2BGR565 = 12
CV_RGB2BGR565 = 13
CV_BGR5652BGR = 14
CV_BGR5652RGB = 15
CV_BGRA2BGR565 = 16
CV_RGBA2BGR565 = 17
CV_BGR5652BGRA = 18
CV_BGR5652RGBA = 19

CV_GRAY2BGR565 = 20
CV_BGR5652GRAY = 21

CV_BGR2BGR555  = 22
CV_RGB2BGR555  = 23
CV_BGR5552BGR  = 24
CV_BGR5552RGB  = 25
CV_BGRA2BGR555 = 26
CV_RGBA2BGR555 = 27
CV_BGR5552BGRA = 28
CV_BGR5552RGBA = 29

CV_GRAY2BGR555 = 30
CV_BGR5552GRAY = 31

CV_BGR2XYZ =    32
CV_RGB2XYZ =    33
CV_XYZ2BGR =    34
CV_XYZ2RGB =    35

CV_BGR2YCrCb =  36
CV_RGB2YCrCb =  37
CV_YCrCb2BGR =  38
CV_YCrCb2RGB =  39

CV_BGR2HSV =    40
CV_RGB2HSV =    41

CV_BGR2Lab =    44
CV_RGB2Lab =    45

CV_BayerBG2BGR = 46
CV_BayerGB2BGR = 47
CV_BayerRG2BGR = 48
CV_BayerGR2BGR = 49

CV_BayerBG2RGB = CV_BayerRG2BGR
CV_BayerGB2RGB = CV_BayerGR2BGR
CV_BayerRG2RGB = CV_BayerBG2BGR
CV_BayerGR2RGB = CV_BayerGB2BGR

CV_BGR2Luv =    50
CV_RGB2Luv =    51
CV_BGR2HLS =    52
CV_RGB2HLS =    53

CV_HSV2BGR =    54
CV_HSV2RGB =    55

CV_Lab2BGR =    56
CV_Lab2RGB =    57
CV_Luv2BGR =    58
CV_Luv2RGB =    59
CV_HLS2BGR =    60
CV_HLS2RGB =    61

CV_COLORCVT_MAX = 100

CV_INTER_NN     = 0
CV_INTER_LINEAR = 1
CV_INTER_CUBIC  = 2
CV_INTER_AREA = 3

CV_WARP_FILL_OUTLIERS = 8
CV_WARP_INVERSE_MAP = 16

CV_SHAPE_RECT = 0
CV_SHAPE_CROSS = 1
CV_SHAPE_ELLIPSE = 2
CV_SHAPE_CUSTOM = 100

CV_MOP_OPEN = 2
CV_MOP_CLOSE = 3
CV_MOP_GRADIENT = 4
CV_MOP_TOPHAT = 5
CV_MOP_BLACKHAT = 6

CV_TM_SQDIFF        = 0
CV_TM_SQDIFF_NORMED = 1
CV_TM_CCORR         = 2
CV_TM_CCORR_NORMED  = 3
CV_TM_CCOEFF        = 4
CV_TM_CCOEFF_NORMED = 5



    ''')

    # functions
    for z in (
        'cvCopyMakeBorder', 'cvSmooth', 'cvFilter2D', 'cvIntegral', 'cvPyrDown', 'cvPyrUp',
        'cvPyrMeanShiftFiltering', 'cvWatershed', 'cvInpaint', 'cvSobel', 'cvLaplace',
        'cvCvtColor', 'cvResize', 'cvWarpAffine', 'cvWarpPerspective',
        'cvRemap', 'cvConvertMaps', 'cvLogPolar', 'cvLinearPolar',
        'cvErode', 'cvDilate', 'cvMorphologyEx', 
        'cvMoments', 'cvGetSpatialMoment', 'cvGetCentralMoment', 'cvGetNormalizedCentralMoment', 'cvGetHuMoments',
        'cvSampleLine', 'cvGetRectSubPix', 'cvGetQuadrangleSubPix',
        'cvMatchTemplate', 
        ):
        mb.free_fun(z).include()
        

    # CvPyramid
    z = mb.class_('CvPyramid')
    z.include()
    z.var('pyramid').exclude()

    # cvCreatePyramid and cvReleasePyramid
    mb.free_fun('cvCreatePyramid').exclude()
    mb.free_fun('cvReleasePyramid').exclude()
    z = mb.free_fun('sdCreatePyramid')
    z.include()
    z.rename('cvCreatePyramid')


    # cvPyrSegmentation
    FT.expose_func(mb.free_fun('cvPyrSegmentation'), ward_indices=(3,), transformer_creators=[
        FT.output_type1('comp')])

    # cvCreateStructuringElementEx
    FT.expose_func(mb.free_fun('cvCreateStructuringElementEx'), ownershiplevel=1, transformer_creators=[
        FT.input_array1d('values')])
    

    # cvReleaseStructuringElement
    z = mb.free_fun('cvReleaseStructuringElement')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.input_double_pointee('element'))


    # cvGetAffineTransform
    FT.expose_func(mb.free_fun('cvGetAffineTransform'), return_arg_index=3)

    # cv2DRotationMatrix
    FT.expose_func(mb.free_fun('cv2DRotationMatrix'), return_arg_index=4)
    
    # cvGetPerspectiveTransform
    FT.expose_func(mb.free_fun('cvGetPerspectiveTransform'), return_arg_index=3)

    # cvCalcEMD2
    FT.expose_func(mb.free_fun('cvCalcEMD2'), return_pointee=False, transformer_creators=[
        FT.distance_function('distance_func', 'userdata')])
    mb.add_doc('cvCalcEMD2', 
        "'distance_func' is a Python function declared as follows:",
        "    def distance_func((int)a, (int)b, (object)userdata) -> (float)x",
        "where",
        "    'a' : the address of a C array of C floats representing the first vector",
        "    'b' : the address of a C array of C floats representing the second vector",
        "    'userdata' : the 'userdata' parameter of cvCalcEMD2()",
        "    'x' : the resultant distance")

    # Contours Retrieving
    cc.write('''
#-----------------------------------------------------------------------------
# Contours Retrieving
#-----------------------------------------------------------------------------


    ''')

    for z in (
        'cvSubstituteContour',
        'cvStartReadChainPoints', 'cvReadChainPoint',
        ):
        mb.free_fun(z).include()


    # TODO: fix these functions
    # cvFindContours

    # cvStartFindContours
    FT.expose_func(mb.free_fun('cvStartFindContours'), ownershiplevel=1, ward_indices=(2,))

    # cvFindNextContour
    FT.expose_func(mb.free_fun('cvFindNextContour'), ward_indices=(1,))

    # cvEndFindContours
    z = mb.free_fun('cvEndFindContours')
    FT.expose_func(z, ward_indices=(1,))
    z.rename('_cvEndFindContours')
    FT.add_underscore(z)
    cc.write('''
def cvEndFindContours(scanner):
    z = _PE._cvEndFindContours(scanner)
    scanner._ownershiplevel = 0 # not owning the structure anymore
    return z
cvEndFindContours.__doc__ = _PE._cvEndFindContours.__doc__
    ''')

    # cvApproxChains
    FT.expose_func(mb.free_fun('cvApproxChains'), ward_indices=(2,))


    # Motion Analysis
    cc.write('''
#-----------------------------------------------------------------------------
# Motion Analysis
#-----------------------------------------------------------------------------


CV_LKFLOW_PYR_A_READY = 1
CV_LKFLOW_PYR_B_READY = 2
CV_LKFLOW_INITIAL_GUESSES = 4
CV_LKFLOW_GET_MIN_EIGENVALS = 8


    ''')

    for z in (
        'cvCalcOpticalFlowLK', 'cvCalcOpticalFlowBM', 'cvCalcOpticalFlowHS',
        'cvEstimateRigidTransform',
        'cvUpdateMotionHistory', 'cvCalcMotionGradient', 'cvCalcGlobalOrientation',
        'cvAcc', 'cvSquareAcc', 'cvMultiplyAcc', 'cvRunningAvg',
        ):
        mb.free_fun(z).include()


    # TODO: fix these functions
    # cvCalcOpticalFlowPyrLK, cvCalcAffineFlowPyrLK

    # cvSegmentMotion
    FT.expose_func(mb.free_fun('cvSegmentMotion'), ward_indices=(3,))
    

    # Object Tracking
    cc.write('''
#-----------------------------------------------------------------------------
# Object Tracking
#-----------------------------------------------------------------------------


    ''')

    for z in (
        'cvCamShift', 'cvMeanShift', 
        'cvConDensUpdateByTime', 'cvConDensInitSampleSet',
        ):
        mb.free_fun(z).include()


    # TODO: fix these functions

    # cvCreateConDensation
    FT.expose_func(mb.free_fun('cvCreateConDensation'), ownershiplevel=1)
    
    # cvReleaseConDensation
    z = mb.free_fun('cvReleaseConDensation')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.input_double_pointee('condens'))

    # cvCreateKalman
    FT.expose_func(mb.free_fun('cvCreateKalman'), ownershiplevel=1)

    # cvReleaseKalman
    z = mb.free_fun('cvReleaseKalman')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.input_double_pointee('kalman'))

    # cvKalmanPredict
    FT.expose_func(mb.free_fun('cvKalmanPredict'), ward_indices=(1,))

    # cvKalmanCorrect
    FT.expose_func(mb.free_fun('cvKalmanCorrect'), ward_indices=(1,))


    # Planar Subdivisions
    cc.write('''
#-----------------------------------------------------------------------------
# Planar Subdivisions
#-----------------------------------------------------------------------------


    ''')

    for z in (
        'cvInitSubdivDelaunay2D',
        'cvCalcSubdivVoronoi2D', 'cvClearSubdivVoronoi2D',
        'cvSubdiv2DNextEdge', 'cvSubdiv2DRotateEdge', 'cvSubdiv2DSymEdge', 'cvSubdiv2DGetEdge',
        'cvTriangleArea',
        ):
        mb.free_fun(z).include()


    # cvCreateSubdiv2D
    FT.expose_func(mb.free_fun('cvCreateSubdiv2D'), ward_indices=(5,))

    # cvCreateSubdivDelaunay2D
    FT.expose_func(mb.free_fun('cvCreateSubdivDelaunay2D'), ward_indices=(2,))

    # cvSubdivDelaunay2DInsert
    FT.expose_func(mb.free_fun('cvSubdivDelaunay2DInsert'), ward_indices=(1,))

    # TODO: fix this function
    # cvSubdiv2DLocate

    # cvFindNearestPoint2D
    FT.expose_func(mb.free_fun('cvFindNearestPoint2D'), ward_indices=(1,))

    # cvSubdiv2DEdge*
    for z in ('cvSubdiv2DEdgeOrg', 'cvSubdiv2DEdgeDst'):
        FT.expose_func(mb.free_fun(z))


    # Contour Processing and Shape Analysis
    cc.write('''
#-----------------------------------------------------------------------------
# Contour Processing and Shape Analysis
#-----------------------------------------------------------------------------


CV_POLY_APPROX_DP = 0

CV_DOMINANT_IPAN = 1

CV_CONTOURS_MATCH_I1 = 1
CV_CONTOURS_MATCH_I2 = 2
CV_CONTOURS_MATCH_I3 = 3

CV_CONTOUR_TREES_MATCH_I1 = 1

CV_CLOCKWISE = 1
CV_COUNTER_CLOCKWISE = 2

CV_COMP_CORREL       = 0
CV_COMP_CHISQR       = 1
CV_COMP_INTERSECT    = 2
CV_COMP_BHATTACHARYYA= 3

    ''')

    for z in (
        'cvContourArea', 'cvMinAreaRect2',
        'cvMatchContourTrees', 'cvCheckContourConvexity',
        'cvFitEllipse2', 'cvMaxRect', 'cvBoxPoints', 'cvPointPolygonTest',
        'cvClearHist', 'cvNormalizeHist', 'cvThreshHist', 'cvCompareHist',
        ):
        mb.free_fun(z).include()

    # TODO: fix these functions
    # cvApproxpoly, cvArcLength, cvContourPerimeter, cvBoundingRect, cvMinEnclosingCircle

    # cvFindDominantPoints
    FT.expose_func(mb.free_fun('cvFindDominantPoints'), ward_indices=(2,))

    # TODO: fix these functions
    # cvMatchshapes

    # cvCreateContourTree
    FT.expose_func(mb.free_fun('cvCreateContourTree'), ward_indices=(2,))

    # cvContourFromContourTree
    FT.expose_func(mb.free_fun('cvContourFromContourTree'), ward_indices=(2,))

    # TODO: fix this function, check with ctypes-opencv
    # cvConvexHull2

    # cvConvexityDefects
    FT.expose_func(mb.free_fun('cvConvexityDefects'), ward_indices=(3,1,2))

    # cvPointSeqFromMat
    FT.expose_func(mb.free_fun('cvPointSeqFromMat'), ward_indices=(3,2))

    # cvMakeHistHeaderForArray, not a safe way to create a histogram, disabled by Minh-Tri Pham

    # cvCreateHist
    FT.expose_func(mb.free_fun('cvCreateHist'), ownershiplevel=1, transformer_creators=[
        FT.input_array2d('ranges')])

    # cvSetHistBinRanges
    FT.expose_func(mb.free_fun('cvSetHistBinRanges'), return_pointee=False, transformer_creators=[
        FT.input_array2d('ranges')])

    # cvReleaseHist
    FT.add_underscore(mb.free_fun('cvReleaseHist'))

    # cvGetMinMaxHistValue
    z = mb.free_fun('cvGetMinMaxHistValue')
    FT.add_underscore(z)
    z._transformer_creators.append(FT.from_address('min_value'))
    z._transformer_creators.append(FT.from_address('max_value'))
    z._transformer_creators.append(FT.from_address('min_idx'))
    z._transformer_creators.append(FT.from_address('max_idx'))
    cc.write('''
def cvGetMinMaxHistValue(hist, return_min_idx=False, return_max_idx=False):
    """(float) min_value, (float) max_value[, (tuple_of_ints)min_idx][, (tuple_of_ints)max_idx] = cvGetMinMaxHistValue((CvHistogram) hist, (bool)return_min_idx=False, (bool)return_max_idx=False)

    Finds the minimum and maximum histogram bins
    [pyopencv] 'min_idx' is returned if 'return_min_idx' is True. 
    [pyopencv] 'max_idx' is returned if 'return_max_idx' is True. 
    """
    min_val = _CT.c_float()
    max_val = _CT.c_float()

    dims = cvGetDims(hist.bins)
    if return_min_idx:
        min_idx = (_CT.c_int*dims)()
        min_addr = _CT.addressof(min_idx)
    else:
        min_addr = 0
    if return_max_idx:
        max_idx = (_CT.c_int*dims)()
        max_addr = _CT.addressof(max_idx)
    else:
        max_addr = 0

    _PE.cvGetMinMaxHistValue(hist, _CT.addressof(min_val), _CT.addressof(max_val), min_addr, max_addr)

    z = (min_val.value, max_val.value)
    if return_min_idx:
        z.append(tuple(min_idx))
    if return_max_idx:
        z.append(tuple(max_idx))
    return z
    ''')

    # cvCopyHist, special case, two transformations
    z = mb.free_fun('cvCopyHist')
    FT.expose_func(z, ownershiplevel=1, transformer_creators=[FT.output_type1('dst')])
    # z.add_transformation(FT.input_double_pointee('dst')) -- wait until requested, buggy though

    # cvCalcBayesianProb
    FT.expose_func(mb.free_fun('cvCalcBayesianProb'), return_pointee=False, transformer_creators=[
        FT.input_array1d('src', 'number'), FT.input_array1d('dst')])

    # cvCalcArrHist and cvCalcHist
    FT.expose_func(mb.free_fun('cvCalcArrHist'), return_pointee=False, transformer_creators=[FT.input_array1d('arr')])
    FT.expose_func(mb.free_fun('cvCalcHist'), return_pointee=False, transformer_creators=[FT.input_array1d('image')])


    # TODO: wrap the rest of cv.h
