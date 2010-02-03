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
    mb.class_('CvFuzzyPoint').include()
    
    # CvFuzzyCurve
    mb.class_('CvFuzzyCurve').include()
    
    # CvFuzzyFunction
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyFunction')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyRule
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyRule')
    z.include()
    z.decls().exclude()
    
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
    z.include_files.append('opencv_extra.hpp')
    mb.init_class(z)
    z.mem_fun('getPointsWithinSphere')._transformer_creators.append(FT.output_std_vector('points'))
    z.constructor(lambda x: len(x.arguments) > 1).exclude()
    z.mem_fun('getNodes').exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::Octree> Octree_init1( bp::tuple const &points, int maxLevels=10, int minPoints=20 )
{
    std::vector<cv::Point3f> points2;
    convert_seq_to_vector(points, points2);
    return boost::shared_ptr<cv::Octree>(new cv::Octree(points2, maxLevels, minPoints ));
}

static bp::sequence sd_getNodes(cv::Octree const &inst) { return convert_vector_to_seq(inst.getNodes()); }
    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&Octree_init1, bp::default_call_policies(), ( bp::arg("points"), bp::arg("maxLevels")=10, bp::arg("maxPoints")=20 )))')
    z.add_registration_code('def( "getNodes", &sd_getNodes)')
    mb.finalize_class(z)
    
    # Mesh3D
    z = mb.class_('Mesh3D')
    mb.init_class(z)
    z.constructor(lambda x: 'vector' in x.decl_string).exclude()
    for t in ('vtx', 'normals'):
        z.var(t).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::Mesh3D> Mesh3D_init1( bp::tuple const &vtx)
{
    std::vector<cv::Point3f> vtx2;
    convert_seq_to_vector(vtx, vtx2);
    return boost::shared_ptr<cv::Mesh3D>(new cv::Mesh3D(vtx2));
}

static bp::sequence get_vtx(cv::Mesh3D const &inst) { return convert_vector_to_seq(inst.vtx); }
static bp::sequence get_normals(cv::Mesh3D const &inst) { return convert_vector_to_seq(inst.normals); }

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&Mesh3D_init1, bp::default_call_policies(), ( bp::arg("vtx") ))  )')
    z.add_registration_code('add_property("vtx", &get_vtx)')
    z.add_registration_code('add_property("normals", &get_normals)')
    for z1 in z.mem_funs('computeNormals'):
        z1._transformer_kwds['alias'] = 'computeNormals'
    mb.finalize_class(z)
    
    # SpinImageModel
    z = mb.class_('SpinImageModel')
    mb.init_class(z)
    z.mem_fun('setLogger').exclude() # wait until requested
    z.mem_fun('match')._transformer_creators.append(FT.output_std_vector_vector('result'))
    for t in ('calcSpinMapCoo', 'geometricConsistency', 'groupingCreteria'):
        z.mem_fun(t).exclude() # wait until requested: not available in OpenCV's Windows package
    z.var('lambda').rename('lambda_') # to avoid a conflict with keyword lambda
    mb.finalize_class(z)
    
    # TickMeter
    mb.class_('TickMeter').include()
    
    # HOGDescriptor
    z = mb.class_('HOGDescriptor')
    z.include_files.append('opencv_extra.hpp')
    # z.include_files.append('_cvaux.h')
    mb.init_class(z)
    z.mem_fun('getDefaultPeopleDetector').exclude()
    z.var('svmDetector').exclude()
    z.add_declaration_code('''
static bp::sequence getDefaultPeopleDetector() {
    return convert_vector_to_seq(cv::HOGDescriptor::getDefaultPeopleDetector());
}

static bp::sequence get_svmDetector(cv::HOGDescriptor const &inst) { return convert_vector_to_seq(inst.svmDetector); }

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
    z.operators().exclude()
    z.add_declaration_code('''
static bp::sequence LDetector_call1( ::cv::LDetector const & inst, bp::object const & image_or_pyr, int maxCount=0, bool scaleCoords=true ){
    std::vector< cv::KeyPoint > keypoints;
    bp::extract<const cv::Mat &> image(image_or_pyr);
    if(image.check()) inst(image(), keypoints, maxCount, scaleCoords);
    else {
        std::vector< cv::Mat > pyr;
        convert_seq_to_vector(image_or_pyr, pyr);
        inst(pyr, keypoints, maxCount, scaleCoords);
    }
    return convert_vector_to_seq(keypoints);
}

    ''')
    z.add_registration_code('def("__call__", &LDetector_call1, (bp::arg("image_or_pyr"), bp::arg("maxCount")=0, bp::arg("scaleCoords")=true))')
    mb.finalize_class(z)
    cc.write('''
YAPE = LDetector
    ''')
    
    # FernClassifier
    # TODO: fix the rest of the member declarations
    z = mb.class_('FernClassifier')
    z.include()
    z.decls().exclude()
    
    # PlanarObjectDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('PlanarObjectDetector')
    z.include()
    z.decls().exclude()
    
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
    z = mb.class_('OneWayDescriptorObject')
    z.include()
    z.decls().exclude()
    
    # LevMarqSparse
    # TODO: fix the rest of the member declarations
    z = mb.class_('LevMarqSparse')
    z.include()
    z.decls().exclude()

    
    #=============================================================================
    # Free Functions
    #=============================================================================

    # TODO:
    # TickMeter's operator <<
    # findOneWayDescriptor
    
    # FAST
    z = mb.free_fun('FAST')
    z.include()
    z._transformer_creators.append(FT.output_std_vector('keypoints'))
