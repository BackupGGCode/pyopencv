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

import common

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cvaux.hpp
#=============================================================================


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # CvCamShiftTracker
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvCamShiftTracker')
    z.include()
    z.decls().exclude()
    
    # CvAdaptiveSkinDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvAdaptiveSkinDetector')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyPoint
    z = mb.class_('CvFuzzyPoint')
    mb.init_class(z)
    # common.register_vec('std::vector', 'CvFuzzyPoint', 'vector_CvFuzzyPoint')    
    # don't register this class because it's used privately only
    mb.finalize_class(z)
    
    # CvFuzzyCurve
    z = mb.class_('CvFuzzyCurve')
    mb.init_class(z)
    common.register_vec('std::vector', 'CvFuzzyCurve', 'vector_CvFuzzyCurve')
    mb.finalize_class(z)
    
    # CvFuzzyFunction
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyFunction')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyRule
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyRule')
    mb.init_class(z)
    z.decls().exclude()
    common.register_vec('std::vector', 'CvFuzzyRule*', 'vector_CvFuzzyRule_Ptr')
    mb.finalize_class(z)
    
    # CvFuzzyController
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyController')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyMeanShiftTracker
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyMeanShiftTracker')
    z.include()
    z.decls().exclude()
    
    # Octree
    z = mb.class_('Octree')
    mb.init_class(z)
    z.mem_fun('getPointsWithinSphere')._transformer_creators.append(FT.arg_output('points'))
    mb.finalize_class(z)

    # Octree::Node
    z = z.class_('Node')
    mb.init_class(z)
    mb.finalize_class(z)
    common.register_vec('std::vector', 'cv::Octree::Node', 'vector_Octree_Node')
        
    # Mesh3D
    z = mb.class_('Mesh3D')
    mb.init_class(z)
    mb.finalize_class(z)
    
    # SpinImageModel
    z = mb.class_('SpinImageModel')
    mb.init_class(z)
    z.mem_fun('setLogger').exclude() # wait until requested
    z.mem_fun('match')._transformer_creators.append(FT.arg_output('result'))
    for t in ('calcSpinMapCoo', 'geometricConsistency', 'groupingCreteria'):
        z.mem_fun(t).exclude() # wait until requested: not available in OpenCV's Windows package
    z.var('lambda').rename('lambda_') # to avoid a conflict with keyword lambda
    mb.finalize_class(z)
    
    # TickMeter
    mb.class_('TickMeter').include()
    
    # HOGDescriptor
    z = mb.class_('HOGDescriptor')
    z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    z.mem_fun('getDefaultPeopleDetector').exclude()
    z.mem_fun('compute')._transformer_creators.append(FT.arg_output('descriptors'))
    for t in ('detect', 'detectMultiScale'):
        z.mem_fun(t)._transformer_creators.append(FT.arg_output('foundLocations'))
    z.var('svmDetector').exclude()
    z.add_declaration_code('''
static cv::Mat getDefaultPeopleDetector() {
    return convert_from_vector_of_T_to_Mat(cv::HOGDescriptor::getDefaultPeopleDetector());
}

static cv::Mat get_svmDetector(cv::HOGDescriptor const &inst) { return convert_from_vector_of_T_to_Mat(inst.svmDetector); }

    ''')
    z.add_registration_code('def("getDefaultPeopleDetector", &::getDefaultPeopleDetector)')
    z.add_registration_code('staticmethod("getDefaultPeopleDetector")')
    z.add_registration_code('add_property("svmDetector", &::get_svmDetector)')    
    mb.finalize_class(z)
    
    # SelfSimDescriptor
    z = mb.class_('SelfSimDescriptor')
    mb.init_class(z)
    mb.finalize_class(z)
    
    # PatchGenerator
    mb.class_('PatchGenerator').include()
    
    # LDetector
    z = mb.class_('LDetector')
    mb.init_class(z)
    for t in z.operators('()'):
        t._transformer_creators.append(FT.arg_output('keypoints'))
    mb.finalize_class(z)
    cc.write('''
YAPE = LDetector
    ''')
    
    # FernClassifier
    z = mb.class_('FernClassifier')
    # z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    for t in z.operators('()'):
        t._transformer_creators.append(FT.arg_output('signature'))
    mb.finalize_class(z)
    
    # FernClassifier::Feature
    z = z.class_('Feature')
    mb.init_class(z)
    mb.finalize_class(z)
    common.register_vec('std::vector', 'cv::FernClassifier::Feature', 'vector_FernClassifier_Feature')
    
    # PlanarObjectDetector
    z = mb.class_('PlanarObjectDetector')
    mb.init_class(z)
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
    mb.finalize_class(z)
    
    # LevMarqSparse
    # TODO: fix the rest of the member declarations
    z = mb.class_('LevMarqSparse')
    z.include()
    z.decls().exclude()
    
    # DefaultRngAuto
    # TODO: fix the rest of the member declarations
    z = mb.class_('DefaultRngAuto')
    z.include()
    z.decls().exclude()
    
    for t in (
        'BackgroundSubtractor', 
        'BackgroundSubtractorMOG', 'CvAffinePose',
        ):
        z = mb.class_(t)
        mb.init_class(z)
        mb.finalize_class(z)
    
    # TODO: BaseKeypoint, CSMatrixGenerator, RandomizedTree, RTreeNode, 
    # RTreeClassifier

    # OneWayDescriptor
    # TODO: fix the rest of the member declarations
    z = mb.class_('OneWayDescriptor')
    z.include()
    z.decls().exclude()
    
    # OneWayDescriptorBase
    # TODO: fix the rest of the member declarations
    z = mb.class_('OneWayDescriptorBase')
    z.include()
    z.decls().exclude()
    
    # OneWayDescriptorObject
    # TODO: fix the rest of the member declarations
    # OpenCV 2.1 bug: this class has not been exposed (lack of CV_EXPORTS)
    # z = mb.class_('OneWayDescriptorObject')
    # z.include()
    # z.decls().exclude()
    
    
    #=============================================================================
    # Free Functions
    #=============================================================================
    
    for t in (
        'find4QuadCornerSubpix',
        ):
        mb.free_fun(t).include()

    # TODO:
    # TickMeter's operator <<
    # findOneWayDescriptor
    
    # FAST
    z = mb.free_fun('FAST')
    z.include()
    z._transformer_creators.append(FT.arg_output('keypoints'))
