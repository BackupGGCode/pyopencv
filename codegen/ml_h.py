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



    ''')

    z = mb.class_('CvVectors')
    z.include()
    FT.expose_member_as_pointee(z, 'next')
    for t in ('ptr', 'fl', 'db', 'data'): # wait until requested
        z.var(t).exclude()


    # TODO: fix member functions with arguments Cv... *
    for z in (
        'CvStatModel', 'CvParamGrid',
        'CvNormalBayesClassifier', 
        ):
        mb.class_(z).include()
        
    # ParamLattice, may or may not be available
    try:
        mb.class_('CvParamLattice').include()
        mb.free_fun('cvParamLattice').include()
        mb.free_fun('cvDefaultParamLattice').include()
    except:
        pass

    # CvKNearest
    z = mb.class_('CvKNearest')
    mb.init_class(z)
    for t in (
        'write_results', 'find_neighbors_direct',
        'find_nearest', # TODO: fix this find_nearest function
        ):
        z.mem_fun(t).exclude()
    mb.finalize_class(z)

    # CvSVMParams
    z = mb.class_('CvSVMParams')
    z.include()
    FT.expose_member_as_pointee(z, 'class_weights')

    # CvSVMKernel # TODO: fix the members of this class
    z = mb.class_('CvSVMKernel')
    z.include()
    z.decls().exclude()

    # CvSVMKernelRow
    z = mb.class_('CvSVMKernelRow')
    for t in ('prev', 'next'):
        FT.expose_member_as_pointee(z, t)
    z.var('data').expose_address = True # wait until requested

    # CvSVMSolver # ToDO: fix this class' members
    z = mb.class_('CvSVMSolver')
    z.include()
    z.decls().exclude()

    # CvSVMDecisionFunc
    z = mb.class_('CvSVMDecisionFunc')
    z.include()
    for t in ('alpha', 'sv_index'):
        FT.expose_member_as_pointee(z, t)

    # CvSVM # TODO: fix this class' members
    z = mb.class_('CvSVM')
    mb.init_class(z)
    z.mem_fun('get_support_vector').exclude() # TODO: fix this function
    mb.finalize_class(z)

    # CvEMParams # TODO: expose this class' members
    z = mb.class_('CvEMParams')
    z.include()
    z.decls().exclude()

    # CvEM
    z = mb.class_('CvEM')
    mb.init_class(z)
    z.mem_fun('get_covs').exclude() # TODO: get_covs()
    mb.finalize_class(z)

    # CvPair16u32s # TODO: expose members
    z = mb.class_('CvPair16u32s')
    z.include()
    z.decls().exclude()

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

    # CvDTreeParams # TODO: expose 'priors'
    z = mb.class_('CvDTreeParams')
    z.include()

    # CvDTreeTrainData
    z = mb.class_('CvDTreeTrainData')
    mb.init_class(z)
    for t in (
        'get_pred_float_buf', 'get_pred_int_buf', 'get_resp_float_buf', 'get_resp_int_buf', 
        'get_cv_lables_buf', 'get_sample_idx_buf',
        ):
        z.mem_fun(t).exclude() # TODO: fix these functions
    for t in (
        'responses_copy', 'cat_count', 'cat_ofs', 'cat_map', 'counts', 'buf', 'direction', 'split_buf', 
        'var_idx', 'var_type', 'priors', 'priors_mult', 'tree_storage', 'temp_storage', 'data_root',
        'node_heap', 'split_heap', 'cv_heap', 'nv_heap',
        ):
        FT.expose_member_as_pointee(z, t)
    mb.finalize_class(z)

    # straightforward classes
    for t in (
        'CvDTree', 'CvForestTree', 'CvRTParams', 'CvRTrees',
        'CvForestERTree', 'CvERTrees',
        'CvBoostParams', 'CvBoostTree', 'CvBoost',
        'CvANN_MLP_TrainParams', 'CvANN_MLP',
        ):
        z = mb.class_(t)
        mb.init_class(z)
        mb.finalize_class(z)

    mb.class_('CvRTrees').mem_fun('get_rng').exclude() # TODO: fix CvRNG first, then fix this get_rng function

    # CvERTreeTrainData
    z = mb.class_('CvERTreeTrainData')
    mb.init_class(z)
    FT.expose_member_as_pointee(z, 'missing_mask')
    mb.finalize_class(z)

    mb.class_('CvANN_MLP').mem_fun('get_weights').exclude() # TODO: fix this func somehow


