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
from pygccxml import declarations as D
from pyplusplus.module_builder import call_policies as CP
import sdpypp
sb = sdpypp.SdModuleBuilder('cv_h', number_of_files=3)

for t in ('CvSubdiv2DPointLocation', 'CvNextEdgeType', 'CvFilter'):
    sb.mb.enums(t).include()

sb.cc.write('''
#=============================================================================
# cvtypes.h
#=============================================================================


# Defines for Distance Transform
CV_DIST_USER    = -1
CV_DIST_L1      = 1
CV_DIST_L2      = 2
CV_DIST_C       = 3
CV_DIST_L12     = 4
CV_DIST_FAIR    = 5
CV_DIST_WELSCH  = 6
CV_DIST_HUBER   = 7

# Haar-like Object Detection structures

CV_HAAR_MAGIC_VAL    = 0x42500000
CV_TYPE_NAME_HAAR    = "opencv-haar-classifier"
CV_HAAR_FEATURE_MAX  = 3

def CV_IS_HAAR_CLASSIFIER(haar_cascade):
    return isinstance(haar_cascade, CvHaarClassifierCascade) and \
        haar_cascade.flags&CV_MAGIC_MASK==CV_HAAR_MAGIC_VAL

''')


z = sb.mb.class_('CvConnectedComp')
sb.init_class(z)
sb.register_vec('std::vector', 'CvConnectedComp', 'vector_CvConnectedComp')
sb.finalize_class(z)
sb.expose_class_Seq('CvConnectedComp')

# cvEndFindContours -- from cv_h_gen.py
z = sb.mb.free_fun('cvEndFindContours')
FT.expose_func(z, ward_indices=(1,))
FT.add_underscore(z)
sb.cc.write('''
def endFindContours(scanner):
    z = _ext._cvEndFindContours(scanner)
    scanner._ownershiplevel = 0 # not owning the structure anymore
    return z
endFindContours.__doc__ = _ext._cvEndFindContours.__doc__
''')

# CvContourScanner
z = sb.mb.class_('_CvContourScanner')
sb.init_class(z)
z.rename('CvContourScanner')
sb.insert_del_interface('CvContourScanner', '_ext._cvEndFindContours')
sb.finalize_class(z)

# CvChainPtReader
z = sb.mb.class_('CvChainPtReader')
sb.init_class(z)
MT.expose_CvSeqReader_members(z, FT)
z.var('deltas').exclude() # wait until requested
sb.finalize_class(z)

# CvContourTree
z = sb.mb.class_('CvContourTree')
sb.init_class(z)
MT.expose_CvSeq_members(z, FT)
sb.finalize_class(z)

#CvConvexityDefect
z = sb.mb.class_('CvConvexityDefect')
sb.init_class(z)
for t in (
    'start', 'end', 'depth_point',
    ):
    FT.expose_member_as_pointee(z, t)
sb.finalize_class(z)

def expose_QuadEdge2D_members(z):
    FT.expose_member_as_array_of_pointees(z, 'pt', 4)
    FT.set_array_item_type_as_size_t(z, 'next')
    
z = sb.mb.class_('CvQuadEdge2D')
sb.init_class(z)
expose_QuadEdge2D_members(z)
sb.finalize_class(z)

# CvSubdiv2DPoint
z = sb.mb.class_('CvSubdiv2DPoint')
sb.init_class(z)
sb.mb.decl('CvSubdiv2DEdge').include()
sb.finalize_class(z)

# CvSubdiv2D
z = sb.mb.class_('CvSubdiv2D')
sb.init_class(z)
MT.expose_CvGraph_members(z, FT)
sb.finalize_class(z)


for z in (
    'CvVect32f', 'CvMatr32f', 'CvVect64d', 'CvMatr64d',
    ):
    sb.mb.decl(z).include()
    
# pointers which are not Cv... * are excluded until further requested
for z in (
    'CvAvgComp',
    ):
    k = sb.mb.class_(z)
    sb.init_class(k)
    for v in k.vars():
        if D.is_pointer(v.type):
            if 'Cv' in v.type.decl_string:
                FT.expose_member_as_pointee(k, v.name)
            else:
                v.exclude()
    sb.finalize_class(k)

# CvHaarFeature
z = sb.mb.class_('CvHaarFeature')
sb.init_class(z)
for t in ('r', 'weight', 'rect'):
    z.decl(t).exclude()
z.add_declaration_code('''
static cv::Rect *CLASS_NAME_get_rect(CvHaarFeature const &inst, int i)
{
    return (cv::Rect *)&inst.rect[i].r;
}

static float CLASS_NAME_get_rect_weight(CvHaarFeature const &inst, int i)
{
    return inst.rect[i].weight;
}

static void CLASS_NAME_set_rect_weight(CvHaarFeature &inst, int i, float _weight)
{
    inst.rect[i].weight = _weight;
}

'''.replace("CLASS_NAME", z.alias))
z.add_registration_code('def("get_rect", &::CLASS_NAME_get_rect, bp::return_internal_reference<>())'.replace("CLASS_NAME", z.alias))
z.add_registration_code('def("get_rect_weight", &::CLASS_NAME_get_rect_weight)'.replace("CLASS_NAME", z.alias))
z.add_registration_code('def("set_rect_weight", &::CLASS_NAME_set_rect_weight)'.replace("CLASS_NAME", z.alias))
sb.finalize_class(z)

# CvHaarClassifier
z = sb.mb.class_('CvHaarClassifier')
sb.init_class(z)
FT.expose_member_as_pointee(z, 'haar_feature')
FT.expose_array_member_as_Mat(z, 'threshold', 'count')
FT.expose_array_member_as_Mat(z, 'left', 'count')
FT.expose_array_member_as_Mat(z, 'right', 'count')
FT.expose_array_member_as_Mat(z, 'alpha', 'count', '1')
sb.finalize_class(z)

# CvHaarStageClassifier
z = sb.mb.class_('CvHaarStageClassifier')
sb.init_class(z)
FT.expose_member_as_array_of_pointees(z, 'classifier', 'inst.count')
sb.finalize_class(z)

# CvHaarClassifierCascade
z = sb.mb.class_('CvHaarClassifierCascade')
sb.init_class(z)
FT.expose_member_as_array_of_pointees(z, 'stage_classifier', 'inst.count')
FT.expose_member_as_pointee(z, 'hid_cascade')
sb.finalize_class(z)

# CvHidHaarFeature
z = sb.mb.class_('CvHidHaarFeature')
sb.init_class(z)
for t in ('p0', 'p1', 'p2', 'p3', 'weight', 'rect'):
    z.decl(t).exclude()
z.add_declaration_code('''
static float CLASS_NAME_get_rect_weight(CvHidHaarFeature const &inst, int i)
{
    return inst.rect[i].weight;
}

static void CLASS_NAME_set_rect_weight(CvHidHaarFeature &inst, int i, float _weight)
{
    inst.rect[i].weight = _weight;
}

'''.replace("CLASS_NAME", z.alias))
z.add_registration_code('def("get_rect_weight", &::CLASS_NAME_get_rect_weight)'.replace("CLASS_NAME", z.alias))
z.add_registration_code('def("set_rect_weight", &::CLASS_NAME_set_rect_weight)'.replace("CLASS_NAME", z.alias))
sb.finalize_class(z)

# CvHidHaarTreeNode
z = sb.mb.class_('CvHidHaarTreeNode')
sb.init_class(z)
sb.finalize_class(z)

# CvHidHaarClassifier
z = sb.mb.class_('CvHidHaarClassifier')
sb.init_class(z)
FT.expose_member_as_array_of_pointees(z, 'node', 'inst.count')
FT.expose_array_member_as_Mat(z, 'alpha', 'count', '1')
sb.finalize_class(z)

# CvHidHaarStageClassifier
z = sb.mb.class_('CvHidHaarStageClassifier')
sb.init_class(z)
FT.expose_member_as_array_of_pointees(z, 'classifier', 'inst.count')
FT.expose_member_as_pointee(z, 'next')
FT.expose_member_as_pointee(z, 'parent')
FT.expose_member_as_pointee(z, 'child')
sb.finalize_class(z)

# CvHidHaarClassifierCascade
z = sb.mb.class_('CvHidHaarClassifierCascade')
sb.init_class(z)
for t in ('pq0', 'pq1', 'pq2', 'pq3', 'p0', 'p1', 'p2', 'p3', 'ipp_stages'):
    z.var(t).exclude()
FT.expose_member_as_array_of_pointees(z, 'stage_classifier', 'inst.count')
sb.finalize_class(z)


sb.cc.write('''
#=============================================================================
# cv.h
#=============================================================================


''')

# Data Structures in cv.h
# pointers which are not Cv... * are excluded until further requested
for t in (
    'CvMSERParams', 
    'CvStarKeypoint',
    'CvPOSITObject',
    ):
    z = sb.mb.class_(t)
    sb.init_class(z)
    try:
        vv = z.vars()
    except RuntimeError:
        vv = []
    for v in vv:
        if D.is_pointer(v.type):
            if 'Cv' in v.type.decl_string:
                FT.expose_member_as_pointee(z, v.name)
            else:
                v.exclude()
    sb.finalize_class(z)


# Image Processing
sb.cc.write('''
#-----------------------------------------------------------------------------
# Image Processing
#-----------------------------------------------------------------------------

    
CV_BLUR_NO_SCALE = 0
CV_BLUR = 1
CV_GAUSSIAN = 2
CV_MEDIAN = 3
CV_BILATERAL = 4

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

CV_WARP_FILL_OUTLIERS = 8
CV_WARP_INVERSE_MAP = 16

CV_SHAPE_RECT = 0
CV_SHAPE_CROSS = 1
CV_SHAPE_ELLIPSE = 2
CV_SHAPE_CUSTOM = 100

CV_MOP_ERODE = 0
CV_MOP_DILATE = 1
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
    'cvSmooth', 'cvPyrMeanShiftFiltering',         
    'cvLogPolar', 'cvLinearPolar',
    'cvSampleLine', 'cvGetQuadrangleSubPix',
    ):
    sb.mb.free_fun(z).include()
    

# cvPyrSegmentation
FT.expose_func(sb.mb.free_fun('cvPyrSegmentation'), ward_indices=(3,), transformer_creators=[
    FT.output_type1('comp', ignore_call_policies=False)])

# cvCalcEMD2
FT.expose_func(sb.mb.free_fun('cvCalcEMD2'), return_pointee=False, transformer_creators=[
    FT.distance_function('distance_func', 'userdata')])
sb.add_doc('calcEMD2', 
    "'distance_func' is a Python function declared as follows:",
    "    def distance_func((int)a, (int)b, (object)userdata) -> (float)x",
    "where",
    "    'a' : the address of a C array of C floats representing the first vector",
    "    'b' : the address of a C array of C floats representing the second vector",
    "    'userdata' : the 'userdata' parameter of cvCalcEMD2()",
    "    'x' : the resultant distance")

# Contours Retrieving
sb.cc.write('''
#-----------------------------------------------------------------------------
# Contours Retrieving
#-----------------------------------------------------------------------------


''')

for z in (
    'cvSubstituteContour',
    'cvStartReadChainPoints', 'cvReadChainPoint',
    ):
    sb.mb.free_fun(z).include()


# cvStartFindContours
FT.expose_func(sb.mb.free_fun('cvStartFindContours'), ownershiplevel=1, ward_indices=(2,))

# cvFindNextContour
FT.expose_func(sb.mb.free_fun('cvFindNextContour'), ward_indices=(1,))

# cvApproxChains
FT.expose_func(sb.mb.free_fun('cvApproxChains'), ward_indices=(2,))


# Motion Analysis
sb.cc.write('''
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
    'cvEstimateRigidTransform', 'cvCalcOpticalFlowFarneback',
    'cvUpdateMotionHistory', 'cvCalcMotionGradient', 'cvCalcGlobalOrientation',
    'cvAcc', 'cvSquareAcc', 'cvMultiplyAcc', 'cvRunningAvg',
    ):
    sb.mb.free_fun(z).include()


# cvCalcOpticalFlowPyrLK
FT.expose_func(sb.mb.free_fun('cvCalcOpticalFlowPyrLK'), return_pointee=False, transformer_creators=[
    FT.input_array1d('prev_features', 'count', output_arrays=[('curr_features','1'), 
        ('status','1'), ('track_error','1')])])

# cvCalcAffineFlowPyrLK
FT.expose_func(sb.mb.free_fun('cvCalcAffineFlowPyrLK'), return_pointee=False, transformer_creators=[
    FT.input_array1d('prev_features', 'count', output_arrays=[('curr_features','1'), 
        ('matrices','1'), ('status','1'), ('track_error','1')])])

# cvSegmentMotion
FT.expose_func(sb.mb.free_fun('cvSegmentMotion'), ward_indices=(3,))


# Planar Subdivisions
sb.cc.write('''
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
    sb.mb.free_fun(z).include()


# cvCreateSubdiv2D
FT.expose_func(sb.mb.free_fun('cvCreateSubdiv2D'), ward_indices=(5,))

# cvCreateSubdivDelaunay2D
FT.expose_func(sb.mb.free_fun('cvCreateSubdivDelaunay2D'), ward_indices=(2,))

# cvSubdivDelaunay2DInsert
FT.expose_func(sb.mb.free_fun('cvSubdivDelaunay2DInsert'), ward_indices=(1,))

# cvSubdiv2DLocate
FT.expose_func(sb.mb.free_fun('cvSubdiv2DLocate'), return_pointee=False, transformer_creators=[
    FT.output_type1('vertex')])

# cvFindNearestPoint2D
FT.expose_func(sb.mb.free_fun('cvFindNearestPoint2D'), ward_indices=(1,))

# cvSubdiv2DEdge*
for z in ('cvSubdiv2DEdgeOrg', 'cvSubdiv2DEdgeDst'):
    FT.expose_func(sb.mb.free_fun(z))


# Contour Processing and Shape Analysis
sb.cc.write('''
#-----------------------------------------------------------------------------
# Contour Processing and Shape Analysis
#-----------------------------------------------------------------------------


CV_POLY_APPROX_DP = 0

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

CV_VALUE = 1
CV_ARRAY = 2

CV_DIST_MASK_3 = 3
CV_DIST_MASK_5 = 5
CV_DIST_MASK_PRECISE = 0

CV_CALIB_CB_FAST_CHECK = 8 # OpenCV 2.1: Equivalent C++ constant not yet available

''')

for z in (
    'cvMatchContourTrees', 
    'cvMaxRect', 'cvBoxPoints', 
    # 'cvClearHist', 'cvNormalizeHist', 'cvThreshHist', 'cvCompareHist', -- disabled by Minh-Tri Pham
    # 'cvCalcProbDensity', 'cvEqualizeHist', -- disabled by Minh-Tri Pham
    ):
    sb.mb.free_fun(z).include()

# cvCreateContourTree
FT.expose_func(sb.mb.free_fun('cvCreateContourTree'), ward_indices=(2,))

# cvContourFromContourTree
FT.expose_func(sb.mb.free_fun('cvContourFromContourTree'), ward_indices=(2,))

# cvConvexityDefects
FT.expose_func(sb.mb.free_fun('cvConvexityDefects'), ward_indices=(3,1,2))

# cvPointSeqFromMat
FT.expose_func(sb.mb.free_fun('cvPointSeqFromMat'), ward_indices=(3,2))

# cvCalcArrBackProject and cvCalcArrBackProjectPatch
# for z in ('cvCalcArrBackProject', 'cvCalcArrBackProjectPatch'):
    # FT.expose_func(sb.mb.free_fun(z), return_pointee=False, transformer_creators=[FT.input_array1d('image')])
# sb.cc.write('''
# backProject = calcArrBackProject
# backProjectPatch = calcArrBackProjectPatch
# ''')


# cvSnakeImage
# FT.expose_func(sb.mb.free_fun('cvSnakeImage'), return_pointee=False, transformer_creators=[
    # FT.input_array1d('alpha'), FT.input_array1d('beta'), FT.input_array1d('gamma')])
sb.mb.free_fun('cvSnakeImage').exclude()
sb.mb.add_declaration_code('''
static void sdSnakeImage( cv::Mat const & image, cv::Mat const & points, bp::object const & alpha, bp::object const & beta, bp::object const & gamma, int coeff_usage, cv::Size const & win, cv::TermCriteria const & criteria, int calc_gradient=1 ){
    char s[500];
    float alpha2, beta2, gamma2;
    std::vector<float> alpha3, beta3, gamma3;
    
    cv::Point *points2; int points2_len; convert_from_Mat_to_array_of_T(points, points2, points2_len);
    
    IplImage img = image;
    
    switch (coeff_usage)
    {
    case CV_VALUE:
        alpha2 = (float) bp::extract<float>(alpha);
        beta2 = (float) bp::extract<float>(beta);
        gamma2 = (float) bp::extract<float>(gamma);
        ::cvSnakeImage(&img, (CvPoint *)points2, points2_len, &alpha2, &beta2, &gamma2, coeff_usage, (CvSize)win, (CvTermCriteria)criteria, calc_gradient);
        break;
    case CV_ARRAY:
        convert_from_object_to_T(alpha, alpha3);
        convert_from_object_to_T(beta, beta3);
        convert_from_object_to_T(gamma, gamma3);
        ::cvSnakeImage(&img, (CvPoint *)points2, points2_len, &alpha3[0], &beta3[0], &gamma3[0], coeff_usage, (CvSize)win, (CvTermCriteria)criteria, calc_gradient);
        break;
    default:
        sprintf(s, "coeff_usage only takes either CV_VALUE or CV_ARRAY as value, %d was given.", coeff_usage);
        PyErr_SetString(PyExc_ValueError, s);
        throw bp::error_already_set(); 
    }
}

''')
sb.mb.add_registration_code('''
    bp::def( 
        "snakeImage"
        , &sdSnakeImage
        , ( bp::arg("image"), bp::arg("points"), bp::arg("alpha"), bp::arg("beta"), bp::arg("gamma"), bp::arg("coeff_usage"), bp::arg("win"), bp::arg("criteria"), bp::arg("calc_gradient")=(int)(1) ) );

''')

# cvDistTransform
FT.expose_func(sb.mb.free_fun('cvDistTransform'), return_pointee=False, transformer_creators=[
    FT.input_array1d('mask', 'mask_size')]) # TODO: worry about the output 'CvArr *labels'



# Feature detection
sb.cc.write('''
#-----------------------------------------------------------------------------
# Feature detection
#-----------------------------------------------------------------------------


''')

# some functions
for z in (
    'cvFindFeatures', 'cvFindFeaturesBoxed',
    'LSHSize', 'cvLSHAdd', 'cvLSHRemove', 'cvLSHQuery',
    # 'cvSURFPoint', 
    'cvStarKeypoint', 
    'cvCheckChessboard',
    ):
    sb.mb.free_fun(z).include()

# CvFeatureTree
sb.mb.class_('CvFeatureTree').include()
sb.insert_del_interface('CvFeatureTree', '_ext._cvReleaseFeatureTree')

# cvCreateKDTree and cvCreateSpillTree
for z in ('cvCreateKDTree', 'cvCreateSpillTree'):
    FT.expose_func(sb.mb.free_fun(z), ownershiplevel=1, ward_indices=(1,))

# cvReleaseFeatureTree
FT.add_underscore(sb.mb.free_fun('cvReleaseFeatureTree'))

# CvLSH
sb.mb.class_('CvLSH').include()
sb.insert_del_interface('CvLSH', '_ext._cvReleaseLSH')

# cvCreateLSH and cvCreateMemoryLSH
for z in ('cvCreateLSH', 'cvCreateMemoryLSH'):
    FT.expose_func(sb.mb.free_fun(z), ownershiplevel=1)

# cvReleaseLSH
FT.add_underscore(sb.mb.free_fun('cvReleaseLSH'))


# CvSURFPoint
z = sb.mb.class_('CvSURFPoint')
sb.init_class(z)
sb.register_vec('std::vector', 'CvSURFPoint', 'vector_CvSURFPoint')
sb.finalize_class(z)
sb.expose_class_Seq('CvSURFPoint')

# CvSURFParams
z = sb.mb.class_('CvSURFParams')
sb.init_class(z)
sb.finalize_class(z)
sb.mb.free_fun('cvSURFParams').include()

# t = sb.register_ti('sdopencv::unbounded_array', ['float'])
# sb.expose_class_Seq(t)

# cvExtractSURF
FT.expose_func(sb.mb.free_fun('cvExtractSURF'), return_pointee=False, transformer_creators=[
    FT.input_as_Seq('CvSURFPoint', 'keypoints', 'storage'), 
    FT.input_CvSeq_ptr_as_vector('float', 'descriptors')])

# POSIT (POSe from ITeration)
sb.cc.write('''
#-----------------------------------------------------------------------------
# POSIT (POSe from ITeration)
#-----------------------------------------------------------------------------


''')

# CvPOSITObject
z = sb.mb.class_('CvPOSITObject')
z.include()
sb.insert_del_interface('CvPOSITObject', '_ext._cvReleasePOSITObject')

z = sb.mb.free_fun('cvReleasePOSITObject')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee(0))

# cvCreatePOSITObject
FT.expose_func(sb.mb.free_fun('cvCreatePOSITObject'), ownershiplevel=1)

# some functions
for t in (
    'cvRANSACUpdateNumIters', 'cvPOSIT',
    'cvTriangulatePoints', 'cvCorrectMatches'):
    sb.mb.free_fun(t).include()

# Camera Calibration, Pose Estimation and Stereo
sb.cc.write('''
#-----------------------------------------------------------------------------
# Camera Calibration, Pose Estimation and Stereo
#-----------------------------------------------------------------------------


''')

# CvStereoBMState
z = sb.mb.class_('CvStereoBMState')
sb.init_class(z)
sb.finalize_class(z)

# Kolmogorov-Zabin stereo-correspondence algorithm (a.k.a. KZ1)
sb.cc.write('''
#-----------------------------------------------------------------------------
# Kolmogorov-Zabin stereo-correspondence algorithm (a.k.a. KZ1)
#-----------------------------------------------------------------------------


''')

# CvStereoGCState
z = sb.mb.class_('CvStereoGCState')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvStereoGCState', '_ext._cvReleaseStereoGCState')

z = sb.mb.free_fun('cvReleaseStereoGCState')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee(0))

# cvCreateStereoGCState
FT.expose_func(sb.mb.free_fun('cvCreateStereoGCState'), ownershiplevel=1)

# some functions
for t in ('cvFindStereoCorrespondenceGC', 'cvReprojectImageTo3D'):
    sb.mb.free_fun(t).include()




sb.register_ti('cv::Point_', ['int'], '_')
sb.register_ti('cv::Point_', ['float'], '_')
sb.register_ti('cv::Rect_', ['int'], '_')
sb.register_ti('cv::Size_', ['int'], '_')
sb.register_ti('cv::TermCriteria')
sb.register_ti('cv::RotatedRect')



sb.done()
