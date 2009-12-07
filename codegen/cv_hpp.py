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
# cv.hpp
#=============================================================================


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # FilterEngine
    # TODO: fix the rest of the member declarations
    z = mb.class_('FilterEngine')
    z.include()
    z.decls().exclude()
    
    # Moments
    z = mb.class_('Moments')
    z.include()
    z.decls(lambda x: 'CvMoments' in x.decl_string).exclude()
    
    # KalmanFilter
    z = mb.class_('KalmanFilter')
    z.include()
    for t in ('predict', 'correct'):
        z.mem_fun(t).call_policies = CP.return_self()
    
    # FeatureEvaluator
    # TODO: fix the rest of the member declarations
    z = mb.class_('FeatureEvaluator')
    z.include()
    z.decls().exclude()
    
    # CascadeClassifier
    # TODO: fix the rest of the member declarations
    z = mb.class_('CascadeClassifier')
    z.include()
    z.decls().exclude()
    
    # StereoBM
    z = mb.class_('StereoBM')
    z.include()
    z.var('state').exclude()
    
    # KeyPoint
    mb.class_('KeyPoint').include()
    mb.class_(lambda x: x.name.startswith('vector<cv::KeyPoint')).exclude()
    
    # SURF
    z = mb.class_('SURF')
    z.include()
    z.operators().exclude()
    z.include_files.append("opencv_extra.hpp")
    z.add_declaration_code('''
static boost::python::tuple call1( ::cv::SURF const & inst, ::cv::Mat const & img, ::cv::Mat const & mask ){
    std::vector<cv::KeyPoint, std::allocator<cv::KeyPoint> > keypoints2;
    inst.operator()(img, mask, keypoints2);
    return convert_vector_to_seq(keypoints2);
}

static boost::python::tuple call2( ::cv::SURF const & inst, ::cv::Mat const & img, ::cv::Mat const & mask, bp::tuple keypoints, bool useProvidedKeypoints=false ){
    std::vector<cv::KeyPoint, std::allocator<cv::KeyPoint> > keypoints2;
    std::vector<float, std::allocator<float> > descriptors2;
    convert_seq_to_vector(keypoints, keypoints2);
    inst.operator()(img, mask, keypoints2, descriptors2, useProvidedKeypoints);
    keypoints = convert_vector_to_seq(keypoints2);
    return bp::make_tuple( keypoints, convert_vector_to_seq(descriptors2) );
}

    ''')
    z.add_registration_code('''def( 
            "__call__"
            , (boost::python::object (*)( ::cv::SURF const &,::cv::Mat const &,::cv::Mat const & ))( &call1 )
            , ( bp::arg("inst"), bp::arg("img"), bp::arg("mask") ) )''')
    z.add_registration_code('''def( 
            "__call__"
            , (boost::python::tuple (*)( ::cv::SURF const &,::cv::Mat const &,::cv::Mat const &,bp::tuple,bool ))( &call2 )
            , ( bp::arg("inst"), bp::arg("img"), bp::arg("mask"), bp::arg("keypoints"), bp::arg("useProvidedKeypoints")=(bool)(false) ) )''')
    mb.class_('CvSURFParams').include()

    
    # MSER
    z = mb.class_('MSER')
    mb.init_class(z)
    z.operators().exclude()
    z.include_files.append("opencv_extra.hpp")
    z.add_declaration_code('''
static boost::python::object call1( ::cv::MSER const & inst, ::cv::Mat & image, ::cv::Mat const & mask ){
    std::vector< std::vector< cv::Point > > msers2;
    inst.operator()(image, msers2, mask);
    return convert_vector_vector_to_seq(msers2);
}

    ''')
    z.add_registration_code('''def( 
            "__call__"
            , (boost::python::object (*)( ::cv::MSER const &,::cv::Mat &,::cv::Mat const & ))( &call1 )
            , ( bp::arg("inst"), bp::arg("image"), bp::arg("mask") ) )''')
    mb.finalize_class(z)
    mb.class_('CvMSERParams').include()
    
    # StarDetector
    z = mb.class_('StarDetector')
    mb.init_class(z)
    z.operators().exclude()
    z.include_files.append("opencv_extra.hpp")
    z.add_declaration_code('''
static boost::python::object call1( ::cv::StarDetector const & inst, ::cv::Mat const & image ){
    std::vector< cv::KeyPoint > keypoints2;
    inst.operator()(image, keypoints2);
    return convert_vector_to_seq(keypoints2);
}

    ''')
    z.add_registration_code('''def( 
            "__call__"
            , (boost::python::object (*)( ::cv::StarDetector const &,::cv::Mat const & ))( &call1 )
            , ( bp::arg("inst"), bp::arg("image") ) )''')
    mb.finalize_class(z)
    mb.class_('CvStarDetectorParams').include()
    
    # CvLevMarq
    # not yet documented, wait until requested: fix the rest of the member declarations
    z = mb.class_('CvLevMarq')
    z.include()
    z.decls().exclude()
    
    # lsh_hash
    mb.class_('lsh_hash').include()
    
    # CvLSHOperations
    # not yet documented, wait until requested: fix the rest of the member declarations
    z = mb.class_('CvLSHOperations')
    z.include()
    z.decls().exclude()
    
    
    #=============================================================================
    # Free functions
    #=============================================================================
    
    # free functions
    for z in (
        'getKernelType', 'getGaussianKernel', 'getDerivKernels', 
        'morphologyDefaultBorderValue', 'getStructuringElement',
        'copyMakeBorder', 'medianBlur', 'GaussianBlur', 'bilateralFilter',
        'boxFilter', 'blur', 'filter2D', 'sepFilter2D', 'Sobel', 'Scharr',
        'Laplacian', 'Canny', 'cornerMinEigenVal', 'cornerHarris', 
        'cornerEigenValsAndVecs', 'preCornerDetect', 'cornerSubPix', 
        'erode', 'dilate', 
        'morphologyEx', 'resize', 'warpAffine', 'warpPerspective', 'remap',
        'convertMaps', 'getRotationMatrix2D', 'invertAffineTransform', 
        'getRectSubPix', 'integral', 'accumulate', 'accumulateSquare', 
        'accumulateProduct', 'accumulateWeighted', 'threshold', 
        'adaptiveThreshold', 'pyrDown', 'pyrUp', 'undistort', 
        'initUndistortRectifyMap', 'getDefaultNewCameraMatrix', 
        'calcOpticalFlowFarneback', 'compareHist', 'equalizeHist', 'watershed',
        'inpaint', 'distanceTransform', 'cvtColor', 'moments', 'matchTemplate',
        'drawContours', 
        'arcLength', 'boundingRect', 'contourArea', 'minAreaRect', 
        'minEnclosingCircle', 'matchShapes', 'isContourConvex', 'fitEllipse',
        'fitLine', 'pointPolygonTest', 'updateMotionHistory', 
        'calcMotionGradient', 'calcGlobalOrientation', 'CamShift', 'meanShift', 
        'estimateAffine3D', 'groupRectangles',
        'Rodrigues', 'RQDecomp3x3', 'decomposeProjectionMatrix', 'matMulDeriv', 
        'composeRT', 'solvePnP', 'initCameraMatrix2D', 'drawChessboardCorners', 
        'calibrationMatrixValues', 'stereoCalibrate', 'stereoRectify', 
        'stereoRectifyUncalibrated', 'reprojectImageTo3D', 
        ):
        mb.free_funs(z).include()

    # TODO:
    # getLinearRowFilter, getLinearColumnFilter, getLinearFilter, createSeparableLinearFilter, createLinearFilter
    # createGaussianFilter,  createDerivFilter, getRowSumFilter, getColumnSumFilter, createBoxFilter
    # getMorphologyRowFilter, getMorphologyColumnFilter, getMorphologyFilter
    # createMorphologyFilter,  , 
    
    # getPerspectiveTransform, getAffineTransform
    for t in ('getPerspectiveTransform', 'getAffineTransform'):
        FT.expose_func(mb.free_fun(t), return_pointee=False, 
            transformer_creators=[FT.input_array1d('src'), FT.input_array1d('dst')])
            
    # goodFeaturesToTrack
    FT.expose_func(mb.free_fun('goodFeaturesToTrack'), return_pointee=False, transformer_creators=[FT.output_std_vector('corners')])

    # 'HoughCircles', 'HoughLines', 'HoughLinesP'
    FT.expose_func(mb.free_fun('HoughCircles'), return_pointee=False, transformer_creators=[FT.output_std_vector('circles')])
    FT.expose_func(mb.free_fun('HoughLines'), return_pointee=False, transformer_creators=[FT.output_std_vector('lines')])
    FT.expose_func(mb.free_fun('HoughLinesP'), return_pointee=False, transformer_creators=[FT.output_std_vector('lines')])
            
    #buildPyramid
    FT.expose_func(mb.free_fun('buildPyramid'), return_pointee=False, transformer_creators=[FT.output_std_vector('dst')])
    
    # calcOpticalFlowPyrLK
    FT.expose_func(mb.free_fun('calcOpticalFlowPyrLK'), return_pointee=False, 
        transformer_creators=[FT.output_std_vector('nextPts'), FT.output_std_vector('status'), 
            FT.output_std_vector('err')])
    
    # TODO
    # calcHist, calcBackProject, floodFill, 
    
    # HuMoments, 
    FT.expose_func(mb.free_fun('HuMoments'), return_pointee=False,
        transformer_creators=[FT.input_array1d('hu')])
        
    # TODO:
    # findContours, , approxPolyDP
    # convexHull, undistortPoints, findHomography
    # projectPoints, 
    
    # findChessboardCorners
    FT.expose_func(mb.free_fun('findChessboardCorners'), return_pointee=False,
        transformer_creators=[FT.output_std_vector('corners')])
        
    # calibrateCamera
    FT.expose_func(mb.free_fun('calibrateCamera'), return_pointee=False,
        transformer_creators=[FT.output_std_vector('rvecs'), FT.output_std_vector('tvecs')])
    
    # TODO: 
    # convertPointsHomogeneous, findFundamentalMat
    
    # computeCorrespondEpilines
    FT.expose_func(mb.free_fun('computeCorrespondEpilines'), return_pointee=False,
        transformer_creators=[FT.output_std_vector('lines')])
    
    
    # TODO:
    # write, read
    
    # wait until requested: missing functions
    # 'estimateRigidTransform', 
    
