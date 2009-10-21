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

