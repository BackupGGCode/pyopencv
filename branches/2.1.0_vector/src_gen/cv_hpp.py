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
# cv.hpp
#=============================================================================


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # BaseRowFilter, BaseColumnFilter, BaseFilter
    for t in ('BaseRowFilter', 'BaseColumnFilter', 'BaseFilter'):
        z = mb.class_(t)
        mb.init_class(z)
        z.operators().exclude() # TODO: expose operators
        mb.finalize_class(z)
        mb.expose_class_Ptr(t, 'cv')
    
    # FilterEngine
    z = mb.class_('FilterEngine')
    mb.init_class(z)
    z.mem_fun('proceed').exclude() # TODO: expose this function
    z.var('rows').exclude() # TODO: expose this variable
    mb.finalize_class(z)
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
    z = mb.class_('FeatureEvaluator')
    mb.init_class(z)
    mb.finalize_class(z)
    mb.expose_class_Ptr('FeatureEvaluator', 'cv')
    
    # CascadeClassifier
    z = mb.class_('CascadeClassifier')
    mb.init_class(z)
    z.mem_fun('detectMultiScale')._transformer_creators.append(FT.arg_output('objects'))
    common.register_ti('CvHaarClassifierCascade')
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
    mb.finalize_class(z)
    
    # CascadeClassifier's sub-classes
    for t in ('DTreeNode', 'DTree', 'Stage'):
        z2 = z.class_(t)
        mb.init_class(z2)
        mb.finalize_class(z2)
        common.register_vec('std::vector', 'cv::CascadeClassifier::'+t, 'vector_CascadeClassifier_'+t)

    
    # StereoBM
    z = mb.class_('StereoBM')
    mb.init_class(z)
    mb.expose_class_Ptr('CvStereoBMState')
    mb.finalize_class(z)
    
    # StereoSGBM
    z = mb.class_('StereoSGBM')
    mb.init_class(z)
    mb.finalize_class(z)
    
    
    # KeyPoint
    z = mb.class_('KeyPoint')
    mb.init_class(z)
    common.register_vec('std::vector', 'cv::KeyPoint')    
    mb.finalize_class(z)
    
    # SURF
    z = mb.class_('SURF')
    mb.init_class(z)
    z.operator(lambda x: len(x.arguments)==3)._transformer_creators.append(FT.arg_output('keypoints'))
    z.operator(lambda x: len(x.arguments)==5)._transformer_creators.append(FT.arg_output('descriptors'))
    mb.finalize_class(z)
    mb.class_('CvSURFParams').include()

    
    # MSER
    z = mb.class_('MSER')
    mb.init_class(z)
    z.operator('()')._transformer_creators.append(FT.arg_output('msers'))
    mb.finalize_class(z)
    mb.class_('CvMSERParams').include()
    
    # StarDetector
    z = mb.class_('StarDetector')
    mb.init_class(z)
    z.operator('()')._transformer_creators.append(FT.arg_output('keypoints'))
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
        'grabCut',
        'inpaint', 'distanceTransform', 'cvtColor', 'moments', 'matchTemplate',
        'drawContours', 
        'arcLength', 'boundingRect', 'contourArea', 'minAreaRect', 
        'minEnclosingCircle', 'matchShapes', 'isContourConvex', 'fitEllipse',
        'fitLine', 'pointPolygonTest', 'updateMotionHistory', 
        'calcMotionGradient', 'calcGlobalOrientation', 'CamShift', 'meanShift', 
        'estimateAffine3D', 
        'Rodrigues', 'RQDecomp3x3', 'decomposeProjectionMatrix', 'matMulDeriv', 
        'composeRT', 'solvePnP', 'initCameraMatrix2D', 'drawChessboardCorners',
        'calibrationMatrixValues', 'stereoCalibrate', 
        'stereoRectifyUncalibrated', 'reprojectImageTo3D', 
        'filterSpeckles', 'getValidDisparityROI', 'validateDisparity',
        ):
        mb.free_funs(z).include()
        
    # FileStorage's read/write functions
    C_to_Python_name_dict = {
        '::std::vector< cv::KeyPoint >': 'list_of_KeyPoint',
        '::cv::SparseMat': 'SparseMat',
        '::cv::MatND': 'MatND',
        '::cv::Mat': 'Mat',
        '::cv::Range': 'Range',
        '::std::string': 'str',
        'double': 'float64',
        'float': 'float32',
        'int': 'int',
        'short int': 'int16',
        '::ushort': 'uint16',
        'short unsigned int': 'uint16',
        '::schar': 'int8',
        'signed char': 'int8',
        '::uchar': 'uint8',
        'unsigned char': 'uint8',
        'bool': 'bool',
    }
        
    # FileStorage's 'write' functions
    for z in mb.free_funs(lambda x: x.name=='write' and \
        x.arguments[0].type.partial_decl_string.startswith('::cv::FileStorage')):
        z.include()
        if len(z.arguments) > 2 and \
            z.arguments[1].type.partial_decl_string.startswith('::std::string'):
            t = D.remove_const(D.remove_reference(z.arguments[2].type))
        else:
            t = D.remove_const(D.remove_reference(z.arguments[1].type))
        name = 'write_'+C_to_Python_name_dict[t.partial_decl_string]
        z._transformer_kwds['alias'] = name
        z.alias = name
    
    # FileNode's 'read' functions
    for z in mb.free_funs(lambda x: x.name=='read' and \
        x.arguments[0].type.partial_decl_string.startswith('::cv::FileNode')):
        z.include()
        if z.arguments[1].name=='keypoints':
            z._transformer_creators.append(FT.arg_output('keypoints'))
        else:
            z._transformer_creators.append(FT.output(z.arguments[1].name))
            FT.doc_output(z, z.arguments[1])   
        t = D.remove_const(D.remove_reference(z.arguments[1].type))
        name = 'read_'+C_to_Python_name_dict[t.partial_decl_string]
        z._transformer_kwds['alias'] = name

    # getPerspectiveTransform, getAffineTransform
    for t in ('getPerspectiveTransform', 'getAffineTransform'):
        FT.expose_func(mb.free_fun(t), return_pointee=False, 
            transformer_creators=[FT.input_array1d('src'), FT.input_array1d('dst')])
            
    # goodFeaturesToTrack
    FT.expose_func(mb.free_fun('goodFeaturesToTrack'), return_pointee=False, 
        transformer_creators=[FT.arg_output('corners')])

    # 'HoughCircles', 'HoughLines', 'HoughLinesP'
    FT.expose_func(mb.free_fun('HoughCircles'), return_pointee=False, 
        transformer_creators=[FT.arg_output('circles')])
    FT.expose_func(mb.free_fun('HoughLines'), return_pointee=False, 
        transformer_creators=[FT.arg_output('lines')])
    FT.expose_func(mb.free_fun('HoughLinesP'), return_pointee=False, 
        transformer_creators=[FT.arg_output('lines')])
    
    # getOptimalNewCameraMatrix
    FT.expose_func(mb.free_fun('getOptimalNewCameraMatrix'), return_pointee=False, 
        transformer_creators=[FT.output_type1('validPixROI')])
    
    # calcHist
    for z in mb.free_funs('calcHist'):
        FT.expose_func(z, return_pointee=False, transformer_creators=[
            FT.input_array1d('images', 'nimages'), FT.input_array1d('channels'),
            FT.input_array1d('histSize', 'dims'), FT.input_array2d('ranges')])
        z._transformer_kwds['alias'] = 'calcHist'
            
    # calcBackProject
    for z in mb.free_funs('calcBackProject'):
        FT.expose_func(z, return_pointee=False, transformer_creators=[
            FT.input_array1d('images', 'nimages'), FT.input_array1d('channels'),
            FT.input_array2d('ranges')])
        z._transformer_kwds['alias'] = 'calcBackProject'
            
    # floodFill
    for z in mb.free_funs('floodFill'):
        FT.expose_func(z, return_pointee=False, transformer_creators=[FT.output_type1('rect')])
        z._transformer_kwds['alias'] = 'floodFill'
    
    # HuMoments
    FT.expose_func(mb.free_fun('HuMoments'), return_pointee=False,
        transformer_creators=[FT.output_static_array('hu', 7)])
        
    # findContours
    z = mb.free_fun(lambda x: x.name=='findContours' and len(x.arguments)==6)
    z.include()
    z._transformer_kwds['alias'] = 'findContours'
    z._transformer_creators.append(FT.arg_output('contours'))
    z._transformer_creators.append(FT.arg_output('hierarchy'))
        
    # findContours
    z = mb.free_fun(lambda x: x.name=='findContours' and len(x.arguments)==5)
    z.include()
    z._transformer_kwds['alias'] = 'findContours'
    z._transformer_creators.append(FT.arg_output('contours'))
    
    # groupRectangles
    for z in mb.free_funs('groupRectangles'):
        z.include()
        z._transformer_kwds['alias'] = 'groupRectangles'
        
    # approxPolyDP    
    for z in mb.free_funs('approxPolyDP'):
        z.include()
        z._transformer_creators.append(FT.arg_output('approxCurve'))
        x = z.arguments[1].type.partial_decl_string
        if 'Point_<int>' in x:
            z._transformer_kwds['alias'] = 'approxPolyDP_int'
        elif 'Point_<float>' in x:
            z._transformer_kwds['alias'] = 'approxPolyDP_float32'
        else:
            z._transformer_kwds['alias'] = 'approxPolyDP' # won't ever occur
        
    # convexHull
    for z in mb.free_funs('convexHull'):
        z.include()
        z._transformer_creators.append(FT.arg_output('hull'))
        x = z.arguments[1].type.partial_decl_string
        if 'Point_<int>' in x:
            z._transformer_kwds['alias'] = 'convexHull_int'
        elif 'Point_<float>' in x:
            z._transformer_kwds['alias'] = 'convexHull_float32'
        else:
            z._transformer_kwds['alias'] = 'convexHullIdx'        
        
    # undistortPoints
    mb.free_funs('undistortPoints').include()
    z = mb.free_fun(lambda x: x.name=='undistortPoints' and 'vector' in x.decl_string)
    z._transformer_kwds['alias'] = 'undistortPoints2'
    z._transformer_creators.append(FT.arg_output('dst'))
        
    # findHomography
    z = mb.free_fun(lambda x: x.name=='findHomography' and len(x.arguments)==4).include()
    z = mb.free_fun(lambda x: x.name=='findHomography' and 'vector' in x.decl_string)
    z.include()
    z._transformer_kwds['alias'] = 'findHomography2'
    z._transformer_creators.append(FT.arg_output('mask'))
        
    # projectPoints
    for z in mb.free_funs('projectPoints'):
        z.include()
        if len(z.arguments) < 10:
            z._transformer_kwds['alias'] = 'projectPoints' 
        else:
            z._transformer_kwds['alias'] = 'projectPoints2'
        z._transformer_creators.append(FT.arg_output('imagePoints'))

    # findChessboardCorners
    FT.expose_func(mb.free_fun('findChessboardCorners'), return_pointee=False,
        transformer_creators=[FT.arg_output('corners')])
    
    # calibrateCamera
    FT.expose_func(mb.free_fun('calibrateCamera'), return_pointee=False,
        transformer_creators=[FT.arg_output('rvecs'), FT.arg_output('tvecs')])
        
    # stereoRectify
    for z in mb.free_funs('stereoRectify'):
        z.include()
        if len(z.arguments)>15:
            z._transformer_kwds['alias'] = 'stereoRectify2'
            z._transformer_creators.extend([FT.output_type1('validPixROI1'), 
                FT.output_type1('validPixROI2')])
        else:
            z._transformer_kwds['alias'] = 'stereoRectify'
    
    # convertPointsHomogeneous
    for z in mb.free_funs('convertPointsHomogeneous'):
        z.include()        
        z._transformer_kwds['alias'] = 'convertPointsHomogeneous3D' if \
            'Point3' in z.decl_string else 'convertPointsHomogeneous2D'
        z._transformer_creators.append(FT.arg_output('dst'))
        
    # findFundamentalMat
    for z in mb.free_funs('findFundamentalMat'):
        z.include()
        if 'vector' in z.decl_string:
            z._transformer_creators.append(FT.arg_output('mask'))
            z._transformer_kwds['alias'] = 'findFundamentalMat2'
    
    # computeCorrespondEpilines
    FT.expose_func(mb.free_fun('computeCorrespondEpilines'), return_pointee=False,
        transformer_creators=[FT.arg_output('lines')])
    
    
