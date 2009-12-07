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
    
    # calcHist
    mb.free_funs('calcHist').exclude()
    mb.add_declaration_code('''
static void sd_calcHist( bp::tuple const & images, bp::tuple const & channels, 
    ::cv::Mat const & mask, bp::object &hist, int dims, bp::tuple const & histSize, 
    bp::tuple const & ranges, bool uniform=true, bool accumulate=false ){
    std::vector< cv::Mat > images2; convert_seq_to_vector(images, images2);
    std::vector< int > channels2; convert_seq_to_vector(channels, channels2);
    std::vector< int > histSize2; convert_seq_to_vector(histSize, histSize2);
    std::vector< std::vector < float > > ranges2; convert_seq_to_vector_vector(ranges, ranges2);
    std::vector< float const * > ranges3;
    ranges3.resize(ranges2.size());
    for(unsigned int i = 0; i < ranges2.size(); ++i ) ranges3[i] = &ranges2[i][0];
    
    bp::extract< ::cv::MatND & > hist_matnd(hist);
    bp::extract< ::cv::SparseMat & > hist_sparsemat(hist);
    
    if(hist_matnd.check())
    {
        cv::MatND &hist_matnd2 = hist_matnd();
        cv::calcHist(&images2[0], images2.size(), &channels2[0], mask,
            hist_matnd2, dims, &histSize2[0], &ranges3[0], uniform, accumulate);
    }
    else if(hist_sparsemat.check())
    {
        cv::SparseMat &hist_sparsemat2 = hist_sparsemat();
        cv::calcHist(&images2[0], images2.size(), &channels2[0], mask,
            hist_sparsemat2, dims, &histSize2[0], &ranges3[0], uniform, accumulate);
    }
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "Only 'MatND' and 'SparseMat' are acceptable types for argument 'hist'.");
        throw bp::error_already_set(); 
    }
}
    ''')
    mb.add_registration_code('''bp::def( 
        "calcHist"
        , (void (*)( bp::tuple const &, bp::tuple const &, ::cv::Mat const &, 
            bp::object &, int, bp::tuple const &, bp::tuple const &, bool, 
            bool ))( &sd_calcHist )
        , ( bp::arg("images"), bp::arg("channels"), bp::arg("mask"), 
            bp::arg("hist"), bp::arg("dims"), bp::arg("histSize"), 
            bp::arg("ranges"), bp::arg("uniform")=bp::object(true), 
            bp::arg("accumulate")=bp::object(false) ) );''')
    
    # calcBackProject
    mb.free_funs('calcBackProject').exclude()
    mb.add_declaration_code('''
static void sd_calcBackProject( bp::tuple const & images, bp::tuple const & channels, 
    bp::object &hist, cv::Mat &backProject, 
    bp::tuple const & ranges, double scale=1, bool uniform=true ){
    std::vector< cv::Mat > images2; convert_seq_to_vector(images, images2);
    std::vector< int > channels2; convert_seq_to_vector(channels, channels2);
    std::vector< std::vector < float > > ranges2; convert_seq_to_vector_vector(ranges, ranges2);
    std::vector< float const * > ranges3;
    ranges3.resize(ranges2.size());
    for(unsigned int i = 0; i < ranges2.size(); ++i ) ranges3[i] = &ranges2[i][0];
    
    bp::extract< ::cv::MatND & > hist_matnd(hist);
    bp::extract< ::cv::SparseMat & > hist_sparsemat(hist);
    
    if(hist_matnd.check())
    {
        cv::MatND &hist_matnd2 = hist_matnd();
        cv::calcBackProject(&images2[0], images2.size(), &channels2[0], 
            hist_matnd2, backProject, &ranges3[0], scale, uniform);
    }
    else if(hist_sparsemat.check())
    {
        cv::SparseMat &hist_sparsemat2 = hist_sparsemat();
        cv::calcBackProject(&images2[0], images2.size(), &channels2[0], 
            hist_sparsemat2, backProject, &ranges3[0], scale, uniform);
    }
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "Only 'MatND' and 'SparseMat' are acceptable types for argument 'hist'.");
        throw bp::error_already_set(); 
    }
}
    ''')
    mb.add_registration_code('''bp::def( 
        "calcBackProject"
        , (void (*)( bp::tuple const &, bp::tuple const &, 
            bp::object &, cv::Mat const &, bp::tuple const &, double, 
            bool ))( &sd_calcBackProject )
        , ( bp::arg("images"), bp::arg("channels"), 
            bp::arg("hist"), bp::arg("backProject"), 
            bp::arg("ranges"), bp::arg("scale")=bp::object(1.0), 
            bp::arg("uniform")=bp::object(true) ) );''')
            
    # floodFill
    mb.free_funs('floodFill').exclude()
    mb.add_declaration_code('''
static bp::tuple sd_floodFill( cv::Mat &image, cv::Point const &seedPoint, 
    cv::Scalar const &newVal, cv::Mat &mask, cv::Scalar const &loDiff, 
    cv::Scalar const &upDiff, int flags) {
    cv::Rect rect;
    int result;
    if(mask.empty()) result = cv::floodFill(image, seedPoint, newVal, &rect, loDiff, upDiff, flags);
    else result = cv::floodFill(image, mask, seedPoint, newVal, &rect, loDiff, upDiff, flags);
    return bp::make_tuple(result, rect);
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "floodFill"
        , (bp::tuple (*)( cv::Mat &, cv::Point const &, cv::Scalar const &, 
            cv::Mat &, cv::Scalar const &, cv::Scalar const &, int ))( &sd_floodFill )
        , ( bp::arg("images"), bp::arg("seedPoint"), 
            bp::arg("newVal"), bp::arg("mask")=bp::object(cv::Mat()), 
            bp::arg("loDiff")=bp::object(cv::Scalar()), 
            bp::arg("upDiff")=bp::object(cv::Scalar()), 
            bp::arg("flags")=bp::object(4) ) );''')
    
    # HuMoments, 
    FT.expose_func(mb.free_fun('HuMoments'), return_pointee=False,
        transformer_creators=[FT.input_array1d('hu')])
        
    # findContours
    mb.free_funs('findContours').exclude()
    mb.add_declaration_code('''
static bp::tuple sd_findContours( cv::Mat const &image, int mode, int method, 
    cv::Point const &offset) {
    std::vector< std::vector< cv::Point > > contours;
    std::vector < cv::Vec4i > hierarchy;
    cv::findContours(image, contours, hierarchy, mode, method, offset);    
    return bp::make_tuple(convert_vector_vector_to_seq(contours), convert_vector_to_seq(hierarchy));
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "findContours"
        , (bp::tuple (*)( cv::Mat &, int, int, cv::Point const & ))( &sd_findContours )
        , ( bp::arg("images"), bp::arg("mode"), bp::arg("method"), 
            bp::arg("offset")=bp::object(cv::Point()) ) );''')
        
    # approxPolyDP
    mb.free_funs('approxPolyDP').exclude()
    mb.add_declaration_code('''
static bp::object sd_approxPolyDP( cv::Mat const &curve, double epsilon, bool closed) {
    std::vector<cv::Point> point2i;
    std::vector<cv::Point2f> point2f;
    bp::object obj;
    if(curve.type() == CV_32SC2) 
    {
        cv::approxPolyDP(curve, point2i, epsilon, closed);
        obj = convert_vector_to_seq(point2i);
    }
    else
    {
        cv::approxPolyDP(curve, point2f, epsilon, closed);
        obj = convert_vector_to_seq(point2f);
    }
    return obj;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "approxPolyDP"
        , (bp::object (*)( cv::Mat const &, double, bool ))( &sd_approxPolyDP )
        , ( bp::arg("curve"), bp::arg("epsilon"), bp::arg("closed") ) );''')
        
    # convexHull
    mb.free_funs('convexHull').exclude()
    mb.add_declaration_code('''
static bp::object sd_convexHullIdx( cv::Mat const &points, bool clockwise=false) {
    std::vector<int> hull;
    cv::convexHull(points, hull, clockwise);
    return convert_vector_to_seq(hull);
}    

static bp::object sd_convexHull( cv::Mat const &points, bool clockwise=false) {
    std::vector<cv::Point> hull2i;
    std::vector<cv::Point2f> hull2f;
    bp::object obj;
    if(points.type() == CV_32SC2)
    {
        cv::convexHull(points, hull2i, clockwise);
        obj = convert_vector_to_seq(hull2i);
    }
    else
    {
        cv::convexHull(points, hull2f, clockwise);
        obj = convert_vector_to_seq(hull2f);
    }
    return obj;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "convexHullIdx"
        , (bp::object (*)( cv::Mat const &, bool ))( &sd_convexHullIdx )
        , ( bp::arg("points"), bp::arg("clockwise")=bp::object(false) ) );''')
    mb.add_registration_code('''bp::def( 
        "convexHull"
        , (bp::object (*)( cv::Mat const &, bool ))( &sd_convexHull )
        , ( bp::arg("points"), bp::arg("clockwise")=bp::object(false) ) );''')
        
    # undistortPoints
    mb.free_funs('undistortPoints').exclude()
    mb.add_declaration_code('''
static cv::Mat sd_undistortPoints( cv::Mat const &src, cv::Mat const &cameraMatrix, 
    cv::Mat const &distCoeffs, cv::Mat const &R, cv::Mat const &P) {
    cv::Mat dst;
    cv::undistortPoints(src, dst, cameraMatrix, distCoeffs, R, P);
    return dst;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "undistortPoints"
        , (bp::object (*)( cv::Mat const &, cv::Mat const &, cv::Mat const &, 
            cv::Mat const &, cv::Mat const & ))( &sd_undistortPoints )
        , ( bp::arg("src"), bp::arg("cameraMatrix"), bp::arg("distCoeffs"),
            bp::arg("R")=bp::object(cv::Mat()), bp::arg("P")=bp::object(cv::Mat()) ) );''')
        
    # findHomography
    mb.free_funs('findHomography').exclude()
    mb.add_declaration_code('''
static cv::Mat sd_findHomography( const Mat& srcPoints, const Mat& dstPoints,
   int method, double ransacReprojThreshold ) {
    cv::Mat mask;
    cv::findHomography(srcPoints, dstPoints, mask, method, ransacReprojThreshold);
    return mask;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "findHomography"
        , (bp::object (*)( cv::Mat const &, cv::Mat const &, int, int ))( &sd_findHomography )
        , ( bp::arg("srcPoints"), bp::arg("dstPoints"), bp::arg("method")=bp::object(0),
            bp::arg("ransacReprojThreshold")=bp::object(0) ) );''')
        
    # TODO:
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
    
