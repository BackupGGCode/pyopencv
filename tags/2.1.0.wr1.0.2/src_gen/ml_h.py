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
# ml.h
#=============================================================================

CV_LOG2PI = (1.8378770664093454835606594728112)

CV_COL_SAMPLE = 0
CV_ROW_SAMPLE = 1

def CV_IS_ROW_SAMPLE(flags):
    return ((flags) & CV_ROW_SAMPLE)

# Variable type
CV_VAR_NUMERICAL    = 0
CV_VAR_ORDERED      = 0
CV_VAR_CATEGORICAL  = 1

CV_TYPE_NAME_ML_SVM         = "opencv-ml-svm"
CV_TYPE_NAME_ML_KNN         = "opencv-ml-knn"
CV_TYPE_NAME_ML_NBAYES      = "opencv-ml-bayesian"
CV_TYPE_NAME_ML_EM          = "opencv-ml-em"
CV_TYPE_NAME_ML_BOOSTING    = "opencv-ml-boost-tree"
CV_TYPE_NAME_ML_TREE        = "opencv-ml-tree"
CV_TYPE_NAME_ML_ANN_MLP     = "opencv-ml-ann-mlp"
CV_TYPE_NAME_ML_CNN         = "opencv-ml-cnn"
CV_TYPE_NAME_ML_RTREES      = "opencv-ml-random-trees"

CV_TRAIN_ERROR  = 0
CV_TEST_ERROR   = 1

# Variable type
CV_VAR_NUMERICAL    = 0
CV_VAR_ORDERED      = 0
CV_VAR_CATEGORICAL  = 1

CV_TYPE_NAME_ML_SVM         = "opencv-ml-svm"
CV_TYPE_NAME_ML_KNN         = "opencv-ml-knn"
CV_TYPE_NAME_ML_NBAYES      = "opencv-ml-bayesian"
CV_TYPE_NAME_ML_EM          = "opencv-ml-em"
CV_TYPE_NAME_ML_BOOSTING    = "opencv-ml-boost-tree"
CV_TYPE_NAME_ML_TREE        = "opencv-ml-tree"
CV_TYPE_NAME_ML_ANN_MLP     = "opencv-ml-ann-mlp"
CV_TYPE_NAME_ML_CNN         = "opencv-ml-cnn"
CV_TYPE_NAME_ML_RTREES      = "opencv-ml-random-trees"

CV_TRAIN_ERROR  = 0
CV_TEST_ERROR   = 1

CV_TS_CONCENTRIC_SPHERES = 0

CV_COUNT     = 0
CV_PORTION   = 1

# StatModel = CvStatModel
# ParamGrid = CvParamGrid
# NormalBayesClassifier = CvNormalBayesClassifier
# KNearest = CvKNearest
# SVMParams = CvSVMParams
# SVMKernel = CvSVMKernel
# SVMSolver = CvSVMSolver
# SVM = CvSVM
# EMParams = CvEMParams
# ExpectationMaximization = CvEM
# DTreeParams = CvDTreeParams
# TrainData = CvMLData
# DecisionTree = CvDTree
# ForestTree = CvForestTree
# RandomTreeParams = CvRTParams
# RandomTrees = CvRTrees
# ERTreeTrainData = CvERTreeTrainData
# ERTree = CvForestERTree
# ERTrees = CvERTrees
# BoostParams = CvBoostParams
# BoostTree = CvBoostTree
# Boost = CvBoost
# ANN_MLP_TrainParams = CvANN_MLP_TrainParams
# NeuralNet_MLP = CvANN_MLP

    ''')

    z = mb.class_('CvVectors')
    z.include()
    FT.expose_member_as_pointee(z, 'next')
    for t in ('ptr', 'fl', 'db', 'data'): # wait until requested
        z.var(t).exclude()

    # ParamLattice, may or may not be available
    try:
        mb.class_('CvParamLattice').include()
        mb.free_fun('cvParamLattice').include()
        mb.free_fun('cvDefaultParamLattice').include()
    except:
        pass
        
    # CvStatModel
    z = mb.class_('CvStatModel')
    mb.init_class(z)    
    mb.finalize_class(z)

    # CvParamGrid
    mb.class_('CvParamGrid').include()
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(min_val=" + repr(self.min_val) + ", max_val=" + repr(self.max_val) \
        + ", step=" + repr(self.step) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", 'CvParamGrid'))
    
    # CvNormalBayesClassifier
    z = mb.class_('CvNormalBayesClassifier')
    mb.init_class(z)
    z.constructors(lambda x: len(x.arguments) > 1).exclude()
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude()
    for t in ('predict', 'train'):
        for t2 in z.mem_funs(t):
            t2._transformer_kwds['alias'] = t
    z.add_wrapper_code('''    
    CvNormalBayesClassifier_wrapper(::cv::Mat const & _train_data, ::cv::Mat const & _responses, ::cv::Mat const & _var_idx=cv::Mat(), ::cv::Mat const & _sample_idx=cv::Mat() )
    : CvNormalBayesClassifier()
      , bp::wrapper< CvNormalBayesClassifier >(){
        // constructor
        train( _train_data, _responses, _var_idx, _sample_idx );
    }
    ''')
    # workaround for the long constructor (their code not yet implemented)
    z.add_registration_code('def( bp::init< cv::Mat const &, cv::Mat const &, bp::optional< cv::Mat const &, cv::Mat const & > >(( bp::arg("_train_data"), bp::arg("_responses"), bp::arg("_var_idx")=cv::Mat(), bp::arg("_sample_idx")=cv::Mat() )) )')
    mb.finalize_class(z)

    # CvKNearest
    z = mb.class_('CvKNearest')
    z.include_files.append('opencv_converters.hpp')
    mb.init_class(z)
    z.constructors(lambda x: 'CvMat' in x.decl_string).exclude()
    for t in ('find_nearest', 'train'):
        z.mem_funs(t).exclude()
    t = z.mem_fun(lambda x: x.name=='train' and not 'CvMat' in x.decl_string)
    t.include()
    t._transformer_kwds['alias'] = 'train'
    # TODO: check if find_nearest() works correctly
    z.add_wrapper_code('''
    bp::object sd_find_nearest( cv::Mat const & _samples, int k, cv::Mat &results, 
        bool return_neighbors_by_addr, cv::Mat &neighbor_responses, cv::Mat &dist ) {
        if(!return_neighbors_by_addr)
            return bp::object(find_nearest((::CvMat const *)get_CvMat_ptr(_samples), k, get_CvMat_ptr(results), 
                0, get_CvMat_ptr(neighbor_responses), get_CvMat_ptr(dist)));
                
        std::vector<int> neighbors2; neighbors2.resize(k*_samples.rows);
        float return_value = find_nearest((::CvMat const *)get_CvMat_ptr(_samples), k, get_CvMat_ptr(results), 
            (const float **)&neighbors2[0], get_CvMat_ptr(neighbor_responses), get_CvMat_ptr(dist));
        return bp::make_tuple(bp::object(return_value), convert_from_T_to_object(neighbors2));
    }
    ''')
    z.add_registration_code('''def("find_nearest", &CvKNearest_wrapper::sd_find_nearest
        , (bp::arg("_samples"), bp::arg("k"), bp::arg("results"), bp::arg("return_neighbors_by_addr")=false, bp::arg("neighbor_response")=cv::Mat(), bp::arg("dist")=cv::Mat() ))''')
    mb.finalize_class(z)

    # CvSVMParams
    z = mb.class_('CvSVMParams')
    z.include()
    z.var('class_weights').exclude()
    z.constructors(lambda x: len(x.arguments) > 1).exclude()
    FT.expose_member_as_Mat(z, 'class_weights')
    z.add_wrapper_code('''
    CvSVMParams_wrapper(int _svm_type, int _kernel_type, double _degree, double _gamma, double _coef0, double _C, double _nu, double _p, cv::Mat const & _class_weights, cv::TermCriteria const &_term_crit )
    : CvSVMParams( _svm_type, _kernel_type, _degree, _gamma, _coef0, _C, _nu, _p, 0, (CvTermCriteria)_term_crit )
      , bp::wrapper< CvSVMParams >(){
        // constructor
        set_class_weights(_class_weights);
    }
    ''')
    z.add_registration_code('def( bp::init< int, int, double, double, double, double, double, double, cv::Mat const &, cv::TermCriteria const & >(( bp::arg("_svm_type"), bp::arg("_kernel_type"), bp::arg("_degree"), bp::arg("_gamma"), bp::arg("_coef0"), bp::arg("_C"), bp::arg("_nu"), bp::arg("_p"), bp::arg("_class_weights"), bp::arg("_term_crit") )) )')

    # CvSVMKernel -- too low-level, wait until requested
    # z = mb.class_('CvSVMKernel')
    # mb.add_doc('CvSVMKernel', "the user must use CvSVMParams to specify the kernel type,",
        # "which defines which Calc function to be used")
    # z.include()
    # z.constructors(lambda x: len(x.arguments) > 1).exclude()
    # z.mem_funs().exclude()
    # z.vars().exclude()
    # z.add_wrapper_code('''
    # CvSVMKernel_wrapper(CvSVMParams const &_params) : CvSVMKernel(&params, 0), bp::wrapper< CvSVMKernel >() {}    
    # bool sd_create(CvSVMParams const &_params) { return CvSVMKernel::create(&params, 0); }    
    # CvSVMParams get_params() { return *params; }
    # ''')
    # z.add_registration_code('def( bp::init< CvSVMParams const & >(( bp::arg("_params") )) )')
    # z.add_registration_code('def( "create", &CvSVMParams_wrapper::sd_create )')
    # z.add_registration_code('add_property( "params", &CvSVMParams_wrapper::get_params )')

    # CvSVMKernelRow -- too low-level, wait until requested
    # z = mb.class_('CvSVMKernelRow')
    # for t in ('prev', 'next'):
        # FT.expose_member_as_pointee(z, t)
    # z.var('data').expose_address = True # wait until requested
    
    # CvSVMSolutionInfo -- too low-level, wait until requested
    # mb.class_('CvSVMSolutionInfo').include()

    # CvSVMSolver -- too low-level, wait until requested

    # CvSVMDecisionFunc -- too low-level, wait until requested
    # z = mb.class_('CvSVMDecisionFunc')
    # z.include()
    # for t in ('alpha', 'sv_index'):
        # FT.expose_member_as_pointee(z, t)

    # CvSVM
    z = mb.class_('CvSVM')
    z.include_files.append( "boost/python/object/life_support.hpp" )
    z.include_files.append( "arrayobject.h" ) # to get NumPy's flags
    z.include_files.append( "ndarray.hpp" )
    mb.init_class(z)
    z.constructors(lambda x: len(x.arguments) > 1).exclude()
    z.mem_funs(lambda t: '::CvMat const *' in t.decl_string).exclude()
    for t in ('train', 'train_auto', 'predict'):
        for t2 in z.mem_funs(lambda x: x.name==t and not 'CvMat' in x.decl_string):
            t2._transformer_kwds['alias'] = t
    z.add_wrapper_code('''    
    CvSVM_wrapper(::cv::Mat const & _train_data, ::cv::Mat const & _responses, ::cv::Mat const & _var_idx=cv::Mat(), ::cv::Mat const & _sample_idx=cv::Mat(), ::CvSVMParams _params=::CvSVMParams( ) )
    : CvSVM()
      , bp::wrapper< CvSVM >(){
        // constructor
        train( _train_data, _responses, _var_idx, _sample_idx, _params );        
    }
      
    bp::object get_support_vector_(int i) {
        int len = get_var_count();
        bp::object result = sdcpp::new_ndarray(1, &len, NPY_FLOAT, 0, 
            (void *)get_support_vector(i), NPY_C_CONTIGUOUS).get_obj();
        bp::objects::make_nurse_and_patient(result.ptr(), bp::object(bp::ptr(this)).ptr());
        return result;
    }
    ''')
    # workaround for the long constructor (their code not yet implemented)
    z.add_registration_code('def( bp::init< cv::Mat const &, cv::Mat const &, bp::optional< cv::Mat const &, cv::Mat const &, CvSVMParams > >(( bp::arg("_train_data"), bp::arg("_responses"), bp::arg("_var_idx")=cv::Mat(), bp::arg("_sample_idx")=cv::Mat(), bp::arg("_params")=::CvSVMParams( ) )) )')
    # get_support_vector
    z.mem_fun('get_support_vector').exclude()
    z.add_registration_code('def( "get_support_vector", &CvSVM_wrapper::get_support_vector_, (bp::arg("i")) )')
    mb.finalize_class(z)

    # CvEMParams 
    z = mb.class_('CvEMParams')
    z.include()
    z.constructor(lambda x: len(x.arguments) > 1).exclude()
    for t in ('probs', 'weights', 'means'):
        FT.expose_member_as_Mat(z, t)
    # wait until requested: 'covs' is turned off for now, the user should use CvEM's member functions instead
    z.var('covs').exclude()
    FT.expose_member_as_TermCriteria(z, 'term_crit')
    # wait until requested, can't expose cv::TermCriteria with a default value
    z.add_wrapper_code('''
    CvEMParams_wrapper(int _nclusters, int _cov_mat_type, int _start_step)
        : CvEMParams(_nclusters, _cov_mat_type, _start_step), bp::wrapper< CvEMParams >() { }
    ''')
    z.add_registration_code('def( bp::init< int, int, int >(( bp::arg("_nclusters"), bp::arg("_cov_mat_type")=1, bp::arg("_start_step")=0 )) )')

    # CvEM
    z = mb.class_('CvEM')
    mb.init_class(z)
    z.constructors(lambda x: 'CvMat' in x.decl_string).exclude()
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude()
    for t in ('train', 'predict'):
        z.mem_fun(lambda x: x.name==t and not 'CvMat' in x.decl_string)._transformer_kwds['alias'] = t
    # wait until requested: enable these functions
    for t in ('get_means', 'get_covs', 'get_weights', 'get_probs'):
        z.mem_fun(t).exclude()
    mb.finalize_class(z)

    # CvPair16u32s # do not expose this old struct
    # z = mb.class_('CvPair16u32s')
    # z.include()
    # z.decls().exclude()

    # CvDTreeSplit
    z = mb.class_('CvDTreeSplit')
    mb.init_class(z)
    FT.expose_member_as_pointee(z, 'next')
    for t in ('subset', 'c', 'split_point'):
        z.var(t).exclude() # TODO: fix these members
    mb.finalize_class(z)

    # CvDTreeNode
    z = mb.class_('CvDTreeNode')
    for t in ('parent', 'left', 'right', 'split'):
        FT.expose_member_as_pointee(z, t)
    for t in ('num_valid', 'cv_Tn', 'cv_node_risk', 'cv_node_error'):
        z.var(t).exclude() # TODO: expose these members

    # CvDTreeParams # TODO: expose 'priors', fix the longer constructor
    z = mb.class_('CvDTreeParams')
    z.include()

    # CvDTreeTrainData
    z = mb.class_('CvDTreeTrainData')
    z.include()
    z.decls().exclude() # TODO: fix this class
    # mb.init_class(z)
    # z.constructors(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these long constructors
    # z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these long member functions
    # for t in (
        # 'responses_copy', 'cat_count', 'cat_ofs', 'cat_map', 'counts', 'buf', 'direction', 'split_buf', 
        # 'var_idx', 'var_type', 'priors', 'priors_mult', 'tree_storage', 'temp_storage', 'data_root',
        # 'node_heap', 'split_heap', 'cv_heap', 'nv_heap', 'rng',
        # ):
        # z.var(t).exclude() # TODO: fix these variables
    # mb.finalize_class(z)
    
    # CvDTree
    z = mb.class_('CvDTree')
    mb.init_class(z)
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    z.mem_fun('calc_error').exclude() # TODO: fix this function
    for t in ('train', 'predict'):
        for t2 in z.mem_funs(t):
            t2._transformer_kwds['alias'] = t
    mb.finalize_class(z)

    # CvForestTree
    z = mb.class_('CvForestTree')
    mb.init_class(z)
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    for t in z.mem_funs('train'):
        t._transformer_kwds['alias'] = 'train'
    mb.finalize_class(z)

    # CvRTParams # TODO: expose 'priors', fix the longer constructor
    z = mb.class_('CvRTParams')
    z.include()

    # CvRTrees
    z = mb.class_('CvRTrees')
    mb.init_class(z)
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    z.mem_fun('calc_error').exclude() # TODO: fix this function
    z.mem_funs(lambda x: 'CvRNG' in x.decl_string).exclude() # TODO: fix these functions
    for t in ('train', 'predict', 'predict_prob'):
        for t2 in z.mem_funs(t):
            t2._transformer_kwds['alias'] = t
    mb.finalize_class(z)

    # CvERTreeTrainData
    z = mb.class_('CvERTreeTrainData')
    z.decls().exclude() # TODO: fix this class

    # CvForestERTree
    z = mb.class_('CvForestERTree')
    mb.init_class(z)
    mb.finalize_class(z)

    # CvERTrees
    z = mb.class_('CvERTrees')
    mb.init_class(z)
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude()
    mb.finalize_class(z)

    # CvBoostParams # TODO: expose 'priors', fix the longer constructor
    z = mb.class_('CvBoostParams')
    z.include()

    # CvBoostTree
    z = mb.class_('CvBoostTree')
    mb.init_class(z)
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    mb.finalize_class(z)

    # CvBoost
    z = mb.class_('CvBoost')
    mb.init_class(z)
    z.constructors(lambda x: 'CvMat' in x.decl_string).exclude()
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    z.mem_fun('calc_error').exclude() # TODO: fix this function
    z.mem_funs(lambda x: 'CvSeq' in x.decl_string).exclude() # TODO: fix these functions
    for t in ('train', 'predict'):
        for t2 in z.mem_funs(t):
            t2._transformer_kwds['alias'] = t
            if t=='predict':
                t2._transformer_creators.append(FT.output_type1('weak_responses'))
    mb.finalize_class(z)

    # CvANN_MLP_TrainParams
    z = mb.class_('CvANN_MLP_TrainParams')
    mb.init_class(z)
    z.decls(lambda x: 'CvTermCriteria' in x.decl_string).exclude() # TODO: fix these declarations
    mb.finalize_class(z)

    # CvANN_MLP
    z = mb.class_('CvANN_MLP')
    mb.init_class(z)
    z.constructors(lambda x: len(x.arguments) > 1).exclude()
    z.mem_funs(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these functions
    for t in ('create', 'train', 'predict'):
        for t2 in z.mem_funs(t):
            t2._transformer_kwds['alias'] = t
    z.mem_fun('get_weights').exclude() # TODO: fix this func somehow
    z.add_wrapper_code('''    
    CvANN_MLP_wrapper(::cv::Mat const & _layer_sizes, int _activ_func=int(::CvANN_MLP::SIGMOID_SYM), double _f_param1=0, double _f_param2=0 )
    : CvANN_MLP()
      , bp::wrapper< CvANN_MLP >(){
        // constructor
        create( _layer_sizes, _activ_func, _f_param1, _f_param2 );
    }
    ''')
    # workaround for the long constructor (their code not yet implemented)
    z.add_registration_code('def( bp::init< cv::Mat const &, bp::optional< int, double, double > >(( bp::arg("_layer_sizes"), bp::arg("_activ_func")=int(::CvANN_MLP::SIGMOID_SYM), bp::arg("_f_param1")=0, bp::arg("_f_param2")=0 )) )')
    mb.finalize_class(z)

    # Convolutional Neural Network, Estimate classifiers algorithms, and Cross validation are not yet enabled

    # Auxilary functions declarations # TODO: fix these functions
    # for z in (
        # 'cvRandMVNormal', 'cvRandGaussMixture', 'cvCreateTestSet',
        # ):
        # mb.free_fun(z).include()

    # TODO: fix these functions
    # cvRandGaussMixture, cvCreateTestSet

    # CvTrainTestSplit # TODO: fix these member variables
    z = mb.class_('CvTrainTestSplit')
    for t in ('train_sample_part', 'count', 'portion', 'class_part'):
        z.vars(t).exclude()

    # CvMLData
    z = mb.class_('CvMLData')
    mb.init_class(z)
    z.decls(lambda x: 'CvMat' in x.decl_string).exclude() # TODO: fix these declarations
    mb.finalize_class(z)

