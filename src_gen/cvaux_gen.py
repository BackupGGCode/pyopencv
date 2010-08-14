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
sb = sdpypp.SdModuleBuilder('cvaux', number_of_files=6)
sb.load_regs('cv_hpp_reg.sdd')



sb.cc.write('''
#=============================================================================
# cvaux.h
#=============================================================================


''')

FT.expose_func(sb.mb.free_fun('cvSegmentImage'), ward_indices=(5,))

# Eigen Objects -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Eigen Objects
#-----------------------------------------------------------------------------


''')

# 1D/2D HMM - TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# 1D/2D HMM
#-----------------------------------------------------------------------------


''')

# A few functions from old stereo gesture recognition demosions - TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# A few functions from old stereo gesture recognition demosions
#-----------------------------------------------------------------------------


''')

# cvCalcImageHomography
FT.expose_func(sb.mb.free_fun('cvCalcImageHomography'), return_pointee=False, transformer_creators=[
    FT.input_static_array('line', 3), FT.input_static_array('intrinsic', 9), FT.output_static_array('homography', 9)])

# Additional operations on Subdivisions -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Additional operations on Subdivisions
#-----------------------------------------------------------------------------


''')

# More operations on sequences -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# More operations on sequences
#-----------------------------------------------------------------------------


CV_DOMINANT_IPAN = 1

''')

# cvFindDominantPoints
FT.expose_func(sb.mb.free_fun('cvFindDominantPoints'), ward_indices=(2,))

# subsections:
# Stereo correspondence -- TODO
sb.cc.write('''
CV_UNDEF_SC_PARAM = 12345

CV_IDP_BIRCHFIELD_PARAM1  = 25    
CV_IDP_BIRCHFIELD_PARAM2  = 5
CV_IDP_BIRCHFIELD_PARAM3  = 12
CV_IDP_BIRCHFIELD_PARAM4  = 15
CV_IDP_BIRCHFIELD_PARAM5  = 25

CV_DISPARITY_BIRCHFIELD  = 0    


''')

sb.mb.free_fun('cvFindStereoCorrespondence').include()

# Epiline functions -- TODO

# Contour Morphing -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Contour Morphing
#-----------------------------------------------------------------------------


''')

# functions -- not in cvaux200.dll.a!!!
# FT.expose_func(sb.mb.free_fun('cvCalcContoursCorrespondence'), 
    # ward_indices=(1,), return_pointee=False, transformer_creators=[
    # FT.input_as_Seq('cv::Point_<int>', 'countour1', 'storage'),
    # FT.input_as_Seq('cv::Point_<int>', 'contour2')])

# FT.expose_func(sb.mb.free_fun('cvMorphContours'), 
    # ward_indices=(1,), return_pointee=False, transformer_creators=[
    # FT.input_as_Seq('cv::Point_<int>', 'countour1', 'storage'),
    # FT.input_as_Seq('cv::Point_<int>', 'contour2'),
    # FT.input_as_Seq('int', 'corr')])



# Texture Descriptors -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Texture Descriptors
#-----------------------------------------------------------------------------


''')

# Face eyes&mouth tracking -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Face eyes&mouth tracking
#-----------------------------------------------------------------------------


''')

# 3D Tracker -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# 3D Tracker
#-----------------------------------------------------------------------------


''')

# Skeletons and Linear-Contour Models -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Skeletons and Linear-Contour Models
#-----------------------------------------------------------------------------


''')

# Background/foreground segmentation -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Background/foreground segmentation
#-----------------------------------------------------------------------------


''')

# Calibration engine -- TODO
sb.cc.write('''
#-----------------------------------------------------------------------------
# Calibration engine
#-----------------------------------------------------------------------------


''')


# Object Tracking
sb.cc.write('''
#-----------------------------------------------------------------------------
# Object Tracking
#-----------------------------------------------------------------------------


''')

# CvConDensation
z = sb.mb.class_('CvConDensation')
sb.init_class(z)
for arg in z.vars():
    if D.is_pointer(arg.type):
        arg.exclude() # wait until requested
sb.finalize_class(z)
sb.insert_del_interface('CvConDensation', '_ext._cvReleaseConDensation')

for z in (
    'cvConDensUpdateByTime', 'cvConDensInitSampleSet',
    ):
    sb.mb.free_fun(z).include()


# cvCreateConDensation
FT.expose_func(sb.mb.free_fun('cvCreateConDensation'), ownershiplevel=1)

# cvReleaseConDensation
z = sb.mb.free_fun('cvReleaseConDensation')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('condens'))




sb.cc.write('''
#=============================================================================
# cvaux.hpp
#=============================================================================


''')

#=============================================================================
# Structures
#=============================================================================

# CvCamShiftTracker
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvCamShiftTracker')
z.include()
z.decls().exclude()

# CvAdaptiveSkinDetector
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvAdaptiveSkinDetector')
z.include()
z.decls().exclude()

# CvFuzzyPoint
z = sb.mb.class_('CvFuzzyPoint')
sb.init_class(z)
# sb.expose_class_vector('CvFuzzyPoint')
# don't expose this class because it's used privately only
sb.finalize_class(z)

# CvFuzzyCurve
z = sb.mb.class_('CvFuzzyCurve')
sb.init_class(z)
sb.expose_class_vector('CvFuzzyCurve')
sb.finalize_class(z)

# CvFuzzyFunction
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvFuzzyFunction')
z.include()
z.decls().exclude()

# CvFuzzyRule
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvFuzzyRule')
sb.init_class(z)
z.decls().exclude()
sb.expose_class_vector('CvFuzzyRule*', 'vector_CvFuzzyRule_Ptr')
sb.finalize_class(z)

# CvFuzzyController
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvFuzzyController')
z.include()
z.decls().exclude()

# CvFuzzyMeanShiftTracker
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvFuzzyMeanShiftTracker')
z.include()
z.decls().exclude()

# Octree
z = sb.mb.class_('Octree')
sb.init_class(z)
z.mem_fun('getPointsWithinSphere')._transformer_creators.append(FT.arg_output('points'))
sb.finalize_class(z)

# Octree::Node
z = z.class_('Node')
sb.init_class(z)
sb.finalize_class(z)
sb.expose_class_vector('cv::Octree::Node', 'vector_Octree_Node')
    
# Mesh3D
z = sb.mb.class_('Mesh3D')
sb.init_class(z)
sb.finalize_class(z)

# SpinImageModel
z = sb.mb.class_('SpinImageModel')
sb.init_class(z)
z.mem_fun('setLogger').exclude() # wait until requested
z.mem_fun('match')._transformer_creators.append(FT.arg_output('result'))
for t in ('calcSpinMapCoo', 'geometricConsistency', 'groupingCreteria'):
    z.mem_fun(t).exclude() # wait until requested: not available in OpenCV's Windows package
z.var('lambda').rename('lambda_') # to avoid a conflict with keyword lambda
sb.finalize_class(z)

# TickMeter
sb.mb.class_('TickMeter').include()

# HOGDescriptor
z = sb.mb.class_('HOGDescriptor')
z.include_files.append('opencv_converters.hpp')
sb.init_class(z)
# z.mem_fun('getDefaultPeopleDetector').exclude()
z.mem_fun('compute')._transformer_creators.append(FT.arg_output('descriptors'))
for t in ('detect', 'detectMultiScale'):
    z.mem_fun(t)._transformer_creators.append(FT.arg_output('foundLocations'))
sb.finalize_class(z)

# SelfSimDescriptor
z = sb.mb.class_('SelfSimDescriptor')
sb.init_class(z)
sb.finalize_class(z)

# PatchGenerator
sb.mb.class_('PatchGenerator').include()

# LDetector
z = sb.mb.class_('LDetector')
sb.init_class(z)
for t in z.operators('()'):
    t._transformer_creators.append(FT.arg_output('keypoints'))
sb.finalize_class(z)
sb.cc.write('''
YAPE = LDetector
''')

# FernClassifier
z = sb.mb.class_('FernClassifier')
# z.include_files.append('opencv_converters.hpp')
sb.init_class(z)
for t in z.operators('()'):
    t._transformer_creators.append(FT.arg_output('signature'))
sb.finalize_class(z)

# FernClassifier::Feature
z = z.class_('Feature')
sb.init_class(z)
sb.finalize_class(z)
sb.expose_class_vector('cv::FernClassifier::Feature', 'vector_FernClassifier_Feature')

# PlanarObjectDetector
z = sb.mb.class_('PlanarObjectDetector')
sb.init_class(z)
z2 = [x for x in z.constructors()]+[x for x in z.mem_funs()]
for t in z2:
    for arg in t.arguments:
        if arg.default_value is not None and ('DEFAULT' in arg.default_value or 'PATCH' in arg.default_value):
            arg.default_value = 'cv::FernClassifier::'+arg.default_value
for t in ('getModelROI', 'getClassifier', 'getDetector'):
    z.mem_fun(t).exclude() # definition not yet available
for z2 in z.operators('()'):
    z2._transformer_creators.append(FT.arg_output('corners'))
    if len(z2.arguments)==5:
        z2._transformer_creators.append(FT.output_type1('pairs'))
sb.finalize_class(z)

# LevMarqSparse
# TODO: fix the rest of the member declarations
z = sb.mb.class_('LevMarqSparse')
z.include()
z.decls().exclude()

# DefaultRngAuto
# TODO: fix the rest of the member declarations
z = sb.mb.class_('DefaultRngAuto')
z.include()
z.decls().exclude()

for t in (
    'BackgroundSubtractor', 
    'BackgroundSubtractorMOG', 'CvAffinePose',
    ):
    z = sb.mb.class_(t)
    sb.init_class(z)
    sb.finalize_class(z)

# TODO: BaseKeypoint, CSMatrixGenerator, RandomizedTree, RTreeNode, 
# RTreeClassifier

# OneWayDescriptor
# TODO: fix the rest of the member declarations
z = sb.mb.class_('OneWayDescriptor')
z.include()
z.decls().exclude()

# OneWayDescriptorBase
# TODO: fix the rest of the member declarations
z = sb.mb.class_('OneWayDescriptorBase')
z.include()
z.decls().exclude()

# OneWayDescriptorObject
# TODO: fix the rest of the member declarations
# OpenCV 2.1 bug: this class has not been exposed (lack of CV_EXPORTS)
# z = sb.mb.class_('OneWayDescriptorObject')
# z.include()
# z.decls().exclude()


#=============================================================================
# Free Functions
#=============================================================================

for t in (
    'find4QuadCornerSubpix',
    ):
    sb.mb.free_fun(t).include()

# TODO:
# TickMeter's operator <<
# findOneWayDescriptor

# FAST
z = sb.mb.free_fun('FAST')
z.include()
z._transformer_creators.append(FT.arg_output('keypoints'))



sb.cc.write('''
#=============================================================================
# cvvidsurf.hpp
#=============================================================================

CV_BLOB_MINW = 5
CV_BLOB_MINH = 5


''')

#=============================================================================
# Structures
#=============================================================================

# CvDefParam
# TODO: fix the rest of the member declarations
z = sb.mb.class_('CvDefParam')
z.include()
FT.expose_member_as_pointee(z, 'next')
for t in ('pName', 'pComment', 'Str'):
    FT.expose_member_as_str(z, t)
for t in ('pDouble', 'pFloat', 'pInt', 'pStr'):
    z.var(t).exclude()

# CvVSModule
z = sb.mb.class_('CvVSModule')
sb.init_class(z)
z.mem_fun('GetModuleName').exclude()
z.add_declaration_code('''    
inline bp::str CvVSModule_GetModuleName(CvVSModule &inst) {  return bp::str(inst.GetModuleName()); }

''')
z.add_registration_code('def("GetModuleName", &::CvVSModule_GetModuleName)')
sb.finalize_class(z)

# CvFGDetector
z = sb.mb.class_('CvFGDetector')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvFGDetector', '_ext._cvReleaseFGDetector')

# cvReleaseFGDetector
z = sb.mb.free_fun('cvReleaseFGDetector')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('ppT'))

# cvCreateFGDetectorBase -- TODO: how to expose 'void *param'?
# FT.expose_func(sb.mb.free_fun('cvCreateFGDetectorBase'), ownershiplevel=1)

# CvBlob
sb.mb.class_('CvBlob').include()
sb.mb.free_fun('cvBlob').include()

# CvBlobSeq
z = sb.mb.class_('CvBlobSeq')
sb.init_class(z)
sb.finalize_class(z)

# CvBlobTrack
z = sb.mb.class_('CvBlobTrack')
z.include()
FT.expose_member_as_pointee(z, 'pBlobSeq')

# CvBlobTrackSeq
z = sb.mb.class_('CvBlobTrackSeq')
sb.init_class(z)
sb.finalize_class(z)

# CvBlobDetector
z = sb.mb.class_('CvBlobDetector')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobDetector', '_ext._cvReleaseBlobDetector')

# cvReleaseBlobDetector
z = sb.mb.free_fun('cvReleaseBlobDetector')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('ppBD'))

# cvCreateBlobDetectorSimple, cvCreateBlobDetectorCC
for t in ('cvCreateBlobDetectorSimple', 'cvCreateBlobDetectorCC'):
    FT.expose_func(sb.mb.free_fun(t), ownershiplevel=1)

# CvBlob
sb.mb.class_('CvDetectedBlob').include()
sb.mb.free_fun('cvDetectedBlob').include()

# CvObjectDetector
z = sb.mb.class_('CvObjectDetector')
sb.init_class(z)
sb.finalize_class(z)

# CvImageDrawer has 2 unimplemented functions, 'Draw' and 'SetShapes'
# Since CvDrawShape is part of CvImageDrawer, the two classes are unexposed
# CvDrawShape
# z = sb.mb.class_('CvDrawShape')
# sb.init_class(z)
# z.decl('shape').exclude() # anonymous enum can't be exposed
# z.add_declaration_code('''
# void CvDrawShape_set_shape(CvDrawShape &inst, int shape) { inst.shape = static_cast<typeof(inst.shape)>(shape); }
# int CvDrawShape_get_shape(CvDrawShape const &inst) { return inst.shape; }
# ''')
# z.add_registration_code('add_property("shape", &::CvDrawShape_get_shape, &::CvDrawShape_set_shape)')
# sb.finalize_class(z)

# CvImageDrawer
# z = sb.mb.class_('CvImageDrawer')
# sb.init_class(z)
# for t in ('Draw', 'GetImage'):
    # FT.expose_func(z.mem_fun(t))
# sb.finalize_class(z)

# CvBlobTrackGen
z = sb.mb.class_('CvBlobTrackGen')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTrackGen', '_ext._cvReleaseBlobTrackGen')

# cvReleaseBlobTrackGen
z = sb.mb.free_fun('cvReleaseBlobTrackGen')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('pBTGen'))

# cvCreateModuleBlobTrackGen1, cvCreateModuleBlobTrackGenYML
for t in ('cvCreateModuleBlobTrackGen1', 'cvCreateModuleBlobTrackGenYML'):
    FT.expose_func(sb.mb.free_fun(t), ownershiplevel=1)

# CvBlobTracker
z = sb.mb.class_('CvBlobTracker')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTracker', '_ext._cvReleaseBlobTracker')

# cvReleaseBlobTracker
z = sb.mb.free_fun('cvReleaseBlobTracker')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('ppT'))

# CvBlobTrackerOne
z = sb.mb.class_('CvBlobTrackerOne')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTrackerOne', '_ext._cvReleaseBlobTrackerOne')

# cvReleaseBlobTrackerOne
z = sb.mb.free_fun('cvReleaseBlobTrackerOne')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('ppT'))

# cvCreateBlobTrackerList -- TODO

sb.cc.write('''
PROFILE_EPANECHNIKOV = 0
PROFILE_DOG = 1

''')

# CvBlobTrackerParamMS
z = sb.mb.class_('CvBlobTrackerParamMS')
sb.init_class(z)
sb.finalize_class(z)

# not yet implemented
# cvCreateBlobTrackerMS1, cvCreateBlobTrackerMS2, and cvCreateBlobTrackerMS1ByList
# for t in ('cvCreateBlobTrackerMS1', 'cvCreateBlobTrackerMS2', 'cvCreateBlobTrackerMS1ByList'):
    # FT.expose_func(sb.mb.free_fun(t), ownershiplevel=1)

# CvBlobTrackerParamLH
z = sb.mb.class_('CvBlobTrackerParamLH')
sb.init_class(z)
sb.finalize_class(z)

# cvCreateBlobTrackerXXX
for t in (
    # 'cvCreateBlobTrackerLHR', 'cvCreateBlobTrackerLHRS', # not yet implemented
    'cvCreateBlobTrackerCC', 'cvCreateBlobTrackerCCMSPF',
    'cvCreateBlobTrackerMSFG', 'cvCreateBlobTrackerMSFGS',
    'cvCreateBlobTrackerMS', 'cvCreateBlobTrackerMSPF'):
    FT.expose_func(sb.mb.free_fun(t), ownershiplevel=1)
    
# CvBlobTrackPostProc
z = sb.mb.class_('CvBlobTrackPostProc')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTrackPostProc', '_ext._cvReleaseBlobTrackPostProc')

# cvReleaseBlobTrackPostProc
z = sb.mb.free_fun('cvReleaseBlobTrackPostProc')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('pBTPP'))

# CvBlobTrackPostProcOne
z = sb.mb.class_('CvBlobTrackPostProcOne')
sb.init_class(z)
sb.finalize_class(z)

# cvCreateBlobTrackPostProcList -- TODO

# cvCreateModuleBlobTrackPostProcXXX
for t in ('cvCreateModuleBlobTrackPostProcKalman',
    'cvCreateModuleBlobTrackPostProcTimeAverRect',
    'cvCreateModuleBlobTrackPostProcTimeAverExp'):
    FT.expose_func(sb.mb.free_fun(t))

# CvBlobTrackPredictor
z = sb.mb.class_('CvBlobTrackPredictor')
sb.init_class(z)
sb.finalize_class(z)

# cvCreateModuleBlobTrackPredictKalman
FT.expose_func(sb.mb.free_fun('cvCreateModuleBlobTrackPredictKalman'))

# CvBlobTrackAnalysis
z = sb.mb.class_('CvBlobTrackAnalysis')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTrackAnalysis', '_ext._cvReleaseBlobTrackAnalysis')

# cvReleaseBlobTrackAnalysis
z = sb.mb.free_fun('cvReleaseBlobTrackAnalysis')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('pBTPP'))

# CvBlobTrackFVGen -- TODO

# CvBlobTrackAnalysis
z = sb.mb.class_('CvBlobTrackAnalysisOne')
sb.init_class(z)
sb.finalize_class(z)

# cvCreateBlobTrackAnalysisList -- TODO

# cvCreateModuleBlobTrackAnalysisXXXX
for t in ('cvCreateModuleBlobTrackAnalysisHistP', 'cvCreateModuleBlobTrackAnalysisHistPV',
    'cvCreateModuleBlobTrackAnalysisHistPVS', 'cvCreateModuleBlobTrackAnalysisHistSS',
    'cvCreateModuleBlobTrackAnalysisTrackDist', 'cvCreateModuleBlobTrackAnalysisIOR',):
    FT.expose_func(sb.mb.free_fun(t))

# CvBlobTrackAnalysisHeight
z = sb.mb.class_('CvBlobTrackAnalysisHeight')
sb.init_class(z)
sb.finalize_class(z)

# CvBlobTrackerAuto
z = sb.mb.class_('CvBlobTrackerAuto')
sb.init_class(z)
sb.finalize_class(z)
sb.insert_del_interface('CvBlobTrackerAuto', '_ext._cvReleaseBlobTrackerAuto')

# cvReleaseBlobTrackerAuto
z = sb.mb.free_fun('cvReleaseBlobTrackerAuto')
FT.add_underscore(z)
z._transformer_creators.append(FT.input_double_pointee('ppT'))

# CvBlobTrackerAutoParam1
z = sb.mb.class_('CvBlobTrackerAutoParam1')
sb.init_class(z)
sb.finalize_class(z)

# cvCreateBlobTrackerAuto1, cvCreateBlobTrackerAuto
# TODO: how to expose 'void *param=NULL'?

# CvTracksTimePos
z = sb.mb.class_('CvTracksTimePos')
sb.init_class(z)
sb.finalize_class(z)

# not yet implemented
# for t in ('cvCreateTracks_One', 'cvCreateTracks_Same', 'cvCreateTracks_AreaErr'):
    # sb.mb.free_fun(t).include()

    
# TODO: expose CvProb and its functions

# TODO: expose CvTestSeq and its functions
    
    

    
#=============================================================================
# Free Functions
#=============================================================================

# TODO:
# cvWriteStruct, cvReadStructByName, 
# findOneWayDescriptor
    
sb.done()
sb.save_regs('cvaux_reg.sdd')
