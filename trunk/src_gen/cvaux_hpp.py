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
    z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    z.mem_fun('getPointsWithinSphere')._transformer_creators.append(FT.arg_std_vector('points', 2))
    z.constructor(lambda x: len(x.arguments) > 1).exclude()
    z.mem_fun('getNodes').exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::Octree> Octree_init1( bp::sequence const &points, int maxLevels=10, int minPoints=20 )
{
    std::vector<cv::Point3f> points2;
    convert_seq_to_vector(points, points2);
    return boost::shared_ptr<cv::Octree>(new cv::Octree(points2, maxLevels, minPoints ));
}

static bp::object sd_getNodes(cv::Octree const &inst) { return convert_from_T_to_object(inst.getNodes()); }
    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&Octree_init1, bp::default_call_policies(), ( bp::arg("points"), bp::arg("maxLevels")=10, bp::arg("maxPoints")=20 )))')
    z.add_registration_code('add_property( "nodes", &sd_getNodes)')
    mb.finalize_class(z)
    
    # Mesh3D
    z = mb.class_('Mesh3D')
    mb.init_class(z)
    z.constructor(lambda x: 'vector' in x.decl_string).exclude()
    for t in ('vtx', 'normals'):
        z.var(t).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::Mesh3D> Mesh3D_init1( bp::sequence const &vtx)
{
    std::vector<cv::Point3f> vtx2;
    convert_seq_to_vector(vtx, vtx2);
    return boost::shared_ptr<cv::Mesh3D>(new cv::Mesh3D(vtx2));
}
    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&Mesh3D_init1, bp::default_call_policies(), ( bp::arg("vtx") ))  )')
    for z1 in z.mem_funs('computeNormals'):
        z1._transformer_kwds['alias'] = 'computeNormals'
    mb.finalize_class(z)
    
    # SpinImageModel
    z = mb.class_('SpinImageModel')
    mb.init_class(z)
    z.mem_fun('setLogger').exclude() # wait until requested
    z.mem_fun('match')._transformer_creators.append(FT.arg_std_vector('result', 2))
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
    z.mem_fun('compute')._transformer_creators.append(FT.arg_std_vector('descriptors', 2))
    for t in ('detect', 'detectMultiScale'):
        z.mem_fun(t)._transformer_creators.append(FT.arg_std_vector('foundLocations', 2))
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
        t._transformer_creators.append(FT.arg_std_vector('keypoints', 2))
    mb.finalize_class(z)
    cc.write('''
YAPE = LDetector
    ''')
    
    # FernClassifier
    z = mb.class_('FernClassifier')
    z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    for t in z.operators('()'):
        t._transformer_creators.append(FT.arg_std_vector('signature', 2))
    z.constructor(lambda x: len(x.arguments) > 5).exclude()
    z.mem_fun('train').exclude()
    z.add_declaration_code('''
static void train( ::cv::FernClassifier & inst, cv::Mat const & points, bp::list const & refimgs, cv::Mat const & labels=convert_from_vector_of_T_to_Mat(std::vector<int>()), int _nclasses=0, int _patchSize=int(::cv::FernClassifier::PATCH_SIZE), int _signatureSize=int(::cv::FernClassifier::DEFAULT_SIGNATURE_SIZE), int _nstructs=int(::cv::FernClassifier::DEFAULT_STRUCTS), int _structSize=int(::cv::FernClassifier::DEFAULT_STRUCT_SIZE), int _nviews=int(::cv::FernClassifier::DEFAULT_VIEWS), int _compressionMethod=int(::cv::FernClassifier::COMPRESSION_NONE), ::cv::PatchGenerator const & patchGenerator=cv::PatchGenerator() ){
    std::vector<cv::Point_<float>, std::allocator<cv::Point_<float> > > points2;
    std::vector<cv::Ptr<cv::Mat>,std::allocator<cv::Ptr<cv::Mat> > > refimgs2;
    std::vector<int, std::allocator<int> > labels2;
    convert_from_Mat_to_vector_of_T(points, points2);
    int i, n = bp::len(refimgs); refimgs2.resize(n);
    for(i = 0; i < n; ++i)
    {
        cv::Mat *obj = new cv::Mat();
        *obj = bp::extract<cv::Mat const &>(refimgs[i]);
        refimgs2[i] = cv::Ptr<cv::Mat>(obj);
    }
    convert_from_Mat_to_vector_of_T(labels, labels2);
    inst.train(points2, refimgs2, labels2, _nclasses, _patchSize, _signatureSize, _nstructs, _structSize, _nviews, _compressionMethod, patchGenerator);
}

static boost::shared_ptr<cv::FernClassifier> FernClassifier_init1( cv::Mat const & points, bp::list const & refimgs, cv::Mat const & labels=convert_from_vector_of_T_to_Mat(std::vector<int>()), int _nclasses=0, int _patchSize=int(::cv::FernClassifier::PATCH_SIZE), int _signatureSize=int(::cv::FernClassifier::DEFAULT_SIGNATURE_SIZE), int _nstructs=int(::cv::FernClassifier::DEFAULT_STRUCTS), int _structSize=int(::cv::FernClassifier::DEFAULT_STRUCT_SIZE), int _nviews=int(::cv::FernClassifier::DEFAULT_VIEWS), int _compressionMethod=int(::cv::FernClassifier::COMPRESSION_NONE), ::cv::PatchGenerator const & patchGenerator=cv::PatchGenerator() ){
    cv::FernClassifier *obj = new cv::FernClassifier();
    train(*obj, points, refimgs, labels, _nclasses, _patchSize, _signatureSize, 
        _nstructs, _structSize, _nviews, _compressionMethod, patchGenerator);
    return boost::shared_ptr<cv::FernClassifier>(obj);
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&FernClassifier_init1, bp::default_call_policies(), ( bp::arg("points"), bp::arg("refimgs"), bp::arg("labels")=cv::Mat(), bp::arg("_nclasses")=0, bp::arg("_patchSize")=int(::cv::FernClassifier::PATCH_SIZE), bp::arg("_signatureSize")=int(::cv::FernClassifier::DEFAULT_SIGNATURE_SIZE), bp::arg("_nstructs")=int(::cv::FernClassifier::DEFAULT_STRUCTS), bp::arg("_structSize")=int(::cv::FernClassifier::DEFAULT_STRUCT_SIZE), bp::arg("_nviews")=int(::cv::FernClassifier::DEFAULT_VIEWS), bp::arg("_compressionMethod")=int(::cv::FernClassifier::COMPRESSION_NONE), bp::arg("patchGenerator")=cv::PatchGenerator() ))  )')
    z.add_registration_code('''def( "train", &train
                , ( bp::arg("inst"), bp::arg("points"), bp::arg("refimgs"), bp::arg("labels")=convert_from_vector_of_T_to_Mat(std::vector<int>()), bp::arg("_nclasses")=(int)(0), bp::arg("_patchSize")=int(::cv::FernClassifier::PATCH_SIZE), bp::arg("_signatureSize")=int(::cv::FernClassifier::DEFAULT_SIGNATURE_SIZE), bp::arg("_nstructs")=int(::cv::FernClassifier::DEFAULT_STRUCTS), bp::arg("_structSize")=int(::cv::FernClassifier::DEFAULT_STRUCT_SIZE), bp::arg("_nviews")=int(::cv::FernClassifier::DEFAULT_VIEWS), bp::arg("_compressionMethod")=int(::cv::FernClassifier::COMPRESSION_NONE), bp::arg("patchGenerator")=cv::PatchGenerator() ) )''')
    mb.finalize_class(z)
    
    # PlanarObjectDetector
    z = mb.class_('PlanarObjectDetector')
    z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    for t in z.mem_funs():
        for arg in t.arguments:
            if arg.default_value is not None and ('DEFAULT' in arg.default_value or 'PATCH' in arg.default_value):
                arg.default_value = 'cv::FernClassifier::'+arg.default_value
    for t in ('getModelROI', 'getClassifier', 'getDetector'):
        z.mem_fun(t).exclude() # definition not yet available
    z.operator(lambda x: x.name=='operator()' and len(x.arguments)==5).exclude() # TODO: fix this operator
    z.constructor(lambda x: len(x.arguments) > 3).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::PlanarObjectDetector> PlanarObjectDetector_init1(bp::list const & pyr, int _npoints=300, int _patchSize=cv::FernClassifier::PATCH_SIZE, int _nstructs=cv::FernClassifier::DEFAULT_STRUCTS, int _structSize=cv::FernClassifier::DEFAULT_STRUCT_SIZE, int _nviews=cv::FernClassifier::DEFAULT_VIEWS, ::cv::LDetector const & detector=cv::LDetector(), ::cv::PatchGenerator const & patchGenerator=cv::PatchGenerator() ){
    std::vector<cv::Mat, std::allocator<cv::Mat> > pyr2;
    convert_from_object_to_T(pyr, pyr2);
    return boost::shared_ptr<cv::PlanarObjectDetector>(
        new cv::PlanarObjectDetector(pyr2, _npoints, _patchSize, _nstructs, _structSize, _nviews, detector, patchGenerator));
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&PlanarObjectDetector_init1, bp::default_call_policies(), ( bp::arg("pyr"), bp::arg("_npoints")=(int)(300), bp::arg("_patchSize")=(int)(cv::FernClassifier::PATCH_SIZE), bp::arg("_nstructs")=(int)(cv::FernClassifier::DEFAULT_STRUCTS), bp::arg("_structSize")=(int)(cv::FernClassifier::DEFAULT_STRUCT_SIZE), bp::arg("_nviews")=(int)(cv::FernClassifier::DEFAULT_VIEWS), bp::arg("detector")=cv::LDetector(), bp::arg("patchGenerator")=cv::PatchGenerator() )) )')
    mb.finalize_class(z)
    
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
    z._transformer_creators.append(FT.arg_std_vector('keypoints', 2))
