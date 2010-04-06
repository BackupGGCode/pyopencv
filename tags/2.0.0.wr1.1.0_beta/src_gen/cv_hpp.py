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
# cv.hpp
#=============================================================================


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # BaseRowFilter, BaseColumnFilter, BaseFilter
    for t in ('BaseRowFilter', 'BaseColumnFilter', 'BaseFilter'):
        z = mb.class_(t)
        # wait until requested: expose the members of the class
        # z.include()
        # z.constructors().exclude()
        # z.operators().exclude()
        mb.expose_class_Ptr(t, 'cv')
    
    # FilterEngine
    # wait until requested: fix the rest of the member declarations
    z = mb.class_('FilterEngine')
    z.include()
    z.decls().exclude()
    mb.expose_class_Ptr('FilterEngine', 'cv')
    
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
    mb.class_('FeatureEvaluator').include()
    mb.expose_class_Ptr('FeatureEvaluator', 'cv')
    
    # CascadeClassifier
    z = mb.class_('CascadeClassifier')
    mb.init_class(z)
    z.mem_fun('detectMultiScale')._transformer_creators.append(FT.arg_std_vector('objects', 2))
    mb.finalize_class(z)
    mb.expose_class_Ptr('CvHaarClassifierCascade')    
    # modify runAt() and setImage() -- I need them able to support old cascade
    z.mem_fun('runAt').exclude()
    z.mem_fun('setImage').exclude()
    z.add_wrapper_code('''
    
    cv::Mat sum, tilted, sqsum;
    CvMat _sum, _sqsum, _tilted;
    
    int my_runAt( cv::Ptr<cv::FeatureEvaluator> &_feval, const cv::Point &pt )
    {
        if( !oldCascade.empty() )
            return cvRunHaarClassifierCascade(oldCascade, pt, 0);
            
        return runAt(_feval, pt);
    }

        
    bool my_setImage( cv::Ptr<cv::FeatureEvaluator> &_feval, const cv::Mat& image )
    {
        if( !oldCascade.empty() )
        {
            sum.create(image.rows+1, image.cols+1, CV_32S);
            tilted.create(image.rows+1, image.cols+1, CV_32S);
            sqsum.create(image.rows+1, image.cols+1, CV_64F);
            cv::integral(image, sum, sqsum, tilted);
            _sum = sum; _sqsum = sqsum; _tilted = tilted;
            cvSetImagesForHaarClassifierCascade( oldCascade, &_sum, &_sqsum, &_tilted, 1. );
            return true;
        }
        
        return setImage(_feval, image);
    }
    
    ''')
    z.add_registration_code('def("runAt", &::CascadeClassifier_wrapper::my_runAt, ( bp::arg("_feval"), bp::arg("pt") ) )')
    z.add_registration_code('def("setImage", &::CascadeClassifier_wrapper::my_setImage, ( bp::arg("_feval"), bp::arg("image") ) )')
    
    # StereoBM
    z = mb.class_('StereoBM')
    z.include()
    mb.expose_class_Ptr('CvStereoBMState')
    
    # KeyPoint
    mb.class_('KeyPoint').include()
    mb.class_(lambda x: x.name.startswith('vector<cv::KeyPoint')).exclude()
    
    # SURF
    z = mb.class_('SURF')
    mb.init_class(z)
    z.operator(lambda x: len(x.arguments)==3)._transformer_creators.append(FT.arg_std_vector('keypoints', 2))
    z.operator(lambda x: len(x.arguments)==5)._transformer_creators.append(FT.arg_std_vector('descriptors', 2))
    mb.finalize_class(z)
    mb.class_('CvSURFParams').include()

    
    # MSER
    z = mb.class_('MSER')
    mb.init_class(z)
    z.operator('()')._transformer_creators.append(FT.arg_std_vector('msers', 2))
    mb.finalize_class(z)
    mb.class_('CvMSERParams').include()
    
    # StarDetector
    z = mb.class_('StarDetector')
    mb.init_class(z)
    z.operator('()')._transformer_creators.append(FT.arg_std_vector('keypoints', 2))
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
        'borderInterpolate', 'getKernelType', 'getLinearRowFilter',
        'getLinearColumnFilter', 'getLinearFilter',
        'createSeparableLinearFilter', 'createLinearFilter',
        'getGaussianKernel', 'createGaussianFilter', 'getDerivKernels', 
        'createDerivFilter', 'getRowSumFilter', 'getColumnSumFilter',
        'createBoxFilter', 'getMorphologyRowFilter', 
        'getMorphologyColumnFilter', 'getMorphologyFilter', 
        'createMorphologyFilter',
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
        'adaptiveThreshold', 'pyrDown', 'pyrUp', 'buildPyramid', 'undistort', 
        'initUndistortRectifyMap', 'getDefaultNewCameraMatrix',
        'calcOpticalFlowPyrLK',
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
        
    # FileStorage's 'write' functions
    for z in mb.free_funs('write'):
        if 'cv::FileStorage' in z.arguments[0].type.decl_string:
            z.include()
            z._transformer_kwds['alias'] = 'write'
    
    # FileNode's 'read' functions
    z = mb.free_fun(lambda x: x.name=='read' and 'cv::FileNode' in x.arguments[0].type.decl_string and x.arguments[1].name=='keypoints')
    z.include()
    z._transformer_creators.append(FT.arg_std_vector('keypoints', 2))
    z._transformer_kwds['alias'] = 'read_KeyPoints'
    read_rename_dict = {
        '::cv::SparseMat &': 'SparseMat',
        '::cv::MatND &': 'MatND',
        '::cv::Mat &': 'Mat',
        '::std::string &': 'str',
        'double &': 'double',
        'float &': 'float',
        'int &': 'inst',
        'short int &': 'short',
        '::ushort &': 'ushort',
        '::schar &': 'schar',
        '::uchar &': 'uchar',
        'bool &': 'bool',
    }
    for elem in read_rename_dict:
        z = mb.free_fun(lambda x: x.name=='read' and 'cv::FileNode' in x.arguments[0].type.decl_string and x.arguments[1].type.decl_string==elem)
        z.include()
        z._transformer_creators.append(FT.output(z.arguments[1].name))
        z._transformer_kwds['alias'] = 'read_'+read_rename_dict[elem]

    # getPerspectiveTransform, getAffineTransform
    for t in ('getPerspectiveTransform', 'getAffineTransform'):
        FT.expose_func(mb.free_fun(t), return_pointee=False, 
            transformer_creators=[FT.input_array1d('src'), FT.input_array1d('dst')])
            
    # goodFeaturesToTrack
    FT.expose_func(mb.free_fun('goodFeaturesToTrack'), return_pointee=False, transformer_creators=[FT.arg_std_vector('corners', 2)])

    # 'HoughCircles', 'HoughLines', 'HoughLinesP'
    FT.expose_func(mb.free_fun('HoughCircles'), return_pointee=False, transformer_creators=[FT.arg_std_vector('circles', 2)])
    FT.expose_func(mb.free_fun('HoughLines'), return_pointee=False, transformer_creators=[FT.arg_std_vector('lines', 2)])
    FT.expose_func(mb.free_fun('HoughLinesP'), return_pointee=False, transformer_creators=[FT.arg_std_vector('lines', 2)])
            
    # calcHist
    mb.free_funs('calcHist').exclude()
    mb.add_declaration_code('''
static void sd_calcHist( bp::list const & images, cv::Mat const & channels, 
    ::cv::Mat const & mask, bp::object &hist, cv::Mat const & histSize, 
    bp::list const & ranges, bool uniform=true, bool accumulate=false ){
    std::vector< cv::Mat > images2; convert_from_object_to_T(images, images2);
    int* channels2; int channels3; convert_from_Mat_to_array_of_T(channels, channels2, channels3);
    int* histSize2; int histSize3; convert_from_Mat_to_array_of_T(histSize, histSize2, histSize3);
    
    int dims = channels3; // dims = minimum between number of channels and number of histogram sizes
    if(histSize3 < dims) dims = histSize3;
    
    std::vector< std::vector < float > > ranges2; convert_from_object_to_T(ranges, ranges2);
    std::vector< float const * > ranges3; ranges3.resize(ranges2.size());
    for(unsigned int i = 0; i < ranges2.size(); ++i ) ranges3[i] = &ranges2[i][0];
    
    bp::extract< ::cv::MatND & > hist_matnd(hist);
    bp::extract< ::cv::SparseMat & > hist_sparsemat(hist);
    
    if(hist_matnd.check())
    {
        cv::MatND &hist_matnd2 = hist_matnd();
        cv::calcHist(&images2[0], images2.size(), channels2, mask,
            hist_matnd2, dims, histSize2, &ranges3[0], uniform, accumulate);
    }
    else if(hist_sparsemat.check())
    {
        cv::SparseMat &hist_sparsemat2 = hist_sparsemat();
        cv::calcHist(&images2[0], images2.size(), channels2, mask,
            hist_sparsemat2, dims, histSize2, &ranges3[0], uniform, accumulate);
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
        , (void (*)( bp::list const &, cv::Mat const &, ::cv::Mat const &, 
            bp::object &, cv::Mat const &, bp::list const &, bool, 
            bool ))( &sd_calcHist )
        , ( bp::arg("images"), bp::arg("channels"), bp::arg("mask"), 
            bp::arg("hist"), bp::arg("histSize"), 
            bp::arg("ranges"), bp::arg("uniform")=bp::object(true), 
            bp::arg("accumulate")=bp::object(false) ) );''')
    
    # calcBackProject
    mb.free_funs('calcBackProject').exclude()
    mb.add_declaration_code('''
static void sd_calcBackProject( bp::list const & images, cv::Mat const & channels, 
    bp::object &hist, cv::Mat &backProject, 
    bp::list const & ranges, double scale=1, bool uniform=true ){
    std::vector< cv::Mat > images2; convert_from_object_to_T(images, images2);
    int* channels2; int channels3; convert_from_Mat_to_array_of_T(channels, channels2, channels3);
    std::vector< std::vector < float > > ranges2; convert_from_object_to_T(ranges, ranges2);
    std::vector< float const * > ranges3;
    ranges3.resize(ranges2.size());
    for(unsigned int i = 0; i < ranges2.size(); ++i ) ranges3[i] = &ranges2[i][0];
    
    bp::extract< ::cv::MatND & > hist_matnd(hist);
    bp::extract< ::cv::SparseMat & > hist_sparsemat(hist);
    
    if(hist_matnd.check())
    {
        cv::MatND &hist_matnd2 = hist_matnd();
        cv::calcBackProject(&images2[0], images2.size(), channels2, 
            hist_matnd2, backProject, &ranges3[0], scale, uniform);
    }
    else if(hist_sparsemat.check())
    {
        cv::SparseMat &hist_sparsemat2 = hist_sparsemat();
        cv::calcBackProject(&images2[0], images2.size(), channels2, 
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
        , (void (*)( bp::list const &, cv::Mat const &, 
            bp::object &, cv::Mat const &, bp::list const &, double, 
            bool ))( &sd_calcBackProject )
        , ( bp::arg("images"), bp::arg("channels"), 
            bp::arg("hist"), bp::arg("backProject"), 
            bp::arg("ranges"), bp::arg("scale")=bp::object(1.0), 
            bp::arg("uniform")=bp::object(true) ) );''')
            
    # floodFill
    for z in mb.free_funs('floodFill'):
        FT.expose_func(z, return_pointee=False, transformer_creators=[FT.output_type1('rect')])
        z._transformer_kwds['alias'] = 'floodFill'
    
    # HuMoments
    FT.expose_func(mb.free_fun('HuMoments'), return_pointee=False,
        transformer_creators=[FT.output_static_array('hu', 7)])
        
    # workaround to fix a bug in OpenCV's internal _findContours
    # this will be removed in the next release of OpenCV
    mb.add_declaration_code('''
namespace cv
{
static void
_findContours( const Mat& image, vector<vector<Point> >& contours,
               vector<Vec4i>* hierarchy, int mode, int method, Point offset )
{
    MemStorage storage(cvCreateMemStorage());
    CvMat _image = image;
    CvSeq* _contours = 0;
    if( hierarchy )
        hierarchy->clear();
    cvFindContours(&_image, storage, &_contours, sizeof(CvContour), mode, method, offset);
    if( !_contours )
    {
        contours.clear();
        return;
    }
    Seq<CvSeq*> all_contours(cvTreeToNodeSeq( _contours, sizeof(CvSeq), storage ));
    size_t i, total = all_contours.size();
    contours.resize(total);
    SeqIterator<CvSeq*> it = all_contours.begin();
    for( i = 0; i < total; i++, ++it )
    {
        CvSeq* c = *it;
        ((CvContour*)c)->color = (int)i;
        Seq<Point>(c).copyTo(contours[i]);
    }

    if( hierarchy )
    {
        hierarchy->resize(total);
        it = all_contours.begin();
        for( i = 0; i < total; i++, ++it )
        {
            CvSeq* c = *it;
            int h_next = c->h_next ? ((CvContour*)c->h_next)->color : -1;
            int h_prev = c->h_prev ? ((CvContour*)c->h_prev)->color : -1;
            int v_next = c->v_next ? ((CvContour*)c->v_next)->color : -1;
            int v_prev = c->v_prev ? ((CvContour*)c->v_prev)->color : -1;
            (*hierarchy)[i] = Vec4i(h_next, h_prev, v_next, v_prev);
        }
    }
}
}

void cv::findContours( const Mat& image, vector<vector<Point> >& contours,
                   vector<Vec4i>& hierarchy, int mode, int method, Point offset )
{
    _findContours(image, contours, &hierarchy, mode, method, offset);
}

void cv::findContours( const Mat& image, vector<vector<Point> >& contours,
                   int mode, int method, Point offset)
{
    _findContours(image, contours, 0, mode, method, offset);
}

    ''')
        
    # findContours
    z = mb.free_fun(lambda x: x.name=='findContours' and len(x.arguments)==6)
    z.include()
    z._transformer_kwds['alias'] = 'findContours'
    z._transformer_creators.append(FT.arg_std_vector('contours', 2))
    z._transformer_creators.append(FT.arg_std_vector('hierarchy', 2))
        
    # approxPolyDP
    mb.free_funs('approxPolyDP').exclude()
    mb.add_declaration_code('''
static cv::Mat sd_approxPolyDP( cv::Mat const &curve, double epsilon, bool closed) {
    cv::Mat approxCurve;
    if(curve.type() == CV_32SC2) 
    {
        std::vector<cv::Point> point2i;
        cv::approxPolyDP(curve, point2i, epsilon, closed);
        convert_from_vector_of_T_to_Mat(point2i, approxCurve);
    }
    else
    {
        std::vector<cv::Point2f> point2f;
        cv::approxPolyDP(curve, point2f, epsilon, closed);
        convert_from_vector_of_T_to_Mat(point2f, approxCurve);
    }
    return approxCurve;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "approxPolyDP"
        , (cv::Mat (*)( cv::Mat const &, double, bool ))( &sd_approxPolyDP )
        , ( bp::arg("curve"), bp::arg("epsilon"), bp::arg("closed") ) );''')
        
    # convexHull
    mb.free_funs('convexHull').exclude()
    z = mb.free_fun(lambda x: x.name=='convexHull' and 'vector<int' in x.arguments[1].type.decl_string)
    z.include()
    z._transformer_kwds['alias'] = 'convexHullIdx'
    z._transformer_creators.append(FT.arg_std_vector('hull', 2))
    
    mb.add_declaration_code('''
static cv::Mat sd_convexHull( cv::Mat const &points, bool clockwise=false) {
    cv::Mat obj;
    if(points.type() == CV_32SC2)
    {
        std::vector<cv::Point> hull2i;
        cv::convexHull(points, hull2i, clockwise);
        convert_from_vector_of_T_to_Mat(hull2i, obj);
    }
    else
    {
        std::vector<cv::Point2f> hull2f;
        cv::convexHull(points, hull2f, clockwise);
        convert_from_vector_of_T_to_Mat(hull2f, obj);
    }
    return obj;
}    
    ''')
    mb.add_registration_code('''bp::def( 
        "convexHull"
        , (cv::Mat (*)( cv::Mat const &, bool ))( &sd_convexHull )
        , ( bp::arg("points"), bp::arg("clockwise")=bp::object(false) ) );''')
        
    # undistortPoints
    mb.free_funs('undistortPoints').include()
    z = mb.free_fun(lambda x: x.name=='undistortPoints' and 'vector' in x.decl_string)
    z._transformer_kwds['alias'] = 'undistortPoints2'
    z._transformer_creators.append(FT.arg_std_vector('dst', 2))
        
    # findHomography
    z = mb.free_fun(lambda x: x.name=='findHomography' and len(x.arguments)==4).include()
    z = mb.free_fun(lambda x: x.name=='findHomography' and 'vector' in x.decl_string)
    z.include()
    z._transformer_kwds['alias'] = 'findHomography2'
    z._transformer_creators.append(FT.arg_std_vector('mask', 2))
        
    # projectPoints
    mb.free_funs('projectPoints').exclude()
    z = mb.free_fun(lambda x: x.name=='projectPoints' and len(x.arguments)==6)
    z._transformer_kwds['alias'] = 'projectPoints'
    FT.expose_func(z, return_pointee=False, transformer_creators=[
        FT.arg_std_vector('imagePoints', 2)])
    z = mb.free_fun(lambda x: x.name=='projectPoints' and len(x.arguments)==12)
    z._transformer_kwds['alias'] = 'projectPoints2'
    FT.expose_func(z, return_pointee=False, transformer_creators=[
        FT.arg_std_vector('imagePoints', 2)])

    # findChessboardCorners
    FT.expose_func(mb.free_fun('findChessboardCorners'), return_pointee=False,
        transformer_creators=[FT.arg_std_vector('corners', 2)])
    
    # calibrateCamera
    FT.expose_func(mb.free_fun('calibrateCamera'), return_pointee=False,
        transformer_creators=[FT.arg_std_vector('rvecs', 2), FT.arg_std_vector('tvecs', 2)])
    
    # convertPointsHomogeneous
    for z in mb.free_funs('convertPointsHomogeneous'):
        z.include()        
        z._transformer_kwds['alias'] = 'convertPointsHomogeneous3D' if 'Point3' in z.decl_string else 'convertPointsHomogeneous2D'
        z._transformer_creators.append(FT.arg_std_vector('dst', 2))
        
    # findFundamentalMat
    for z in mb.free_funs('findFundamentalMat'):
        z.include()
        if 'vector' in z.decl_string:
            z._transformer_creators.append(FT.arg_std_vector('mask', 2))
            z._transformer_kwds['alias'] = 'findFundamentalMat2'
    
    # computeCorrespondEpilines
    FT.expose_func(mb.free_fun('computeCorrespondEpilines'), return_pointee=False,
        transformer_creators=[FT.arg_std_vector('lines', 2)])
    
    
