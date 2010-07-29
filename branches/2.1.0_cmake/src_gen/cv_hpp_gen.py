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


import function_transformers as FT
import memvar_transformers as MT
from pygccxml import declarations as D
from pyplusplus.module_builder import call_policies as CP
import sdpypp
sb = sdpypp.SdModuleBuilder('cv_hpp', number_of_files=3)

sb.mb.enums(lambda x: x.parent.name=="cv").include()

sb.cc.write('''
#=============================================================================
# cv.hpp
#=============================================================================


''')

#=============================================================================
# Structures
#=============================================================================

# BaseRowFilter, BaseColumnFilter, BaseFilter
for t in ('BaseRowFilter', 'BaseColumnFilter', 'BaseFilter'):
    z = sb.mb.class_(t)
    sb.init_class(z)
    z.operators().exclude() # TODO: expose operators
    sb.finalize_class(z)
    sb.expose_class_Ptr(t, 'cv')

# FilterEngine
z = sb.mb.class_('FilterEngine')
sb.init_class(z)
z.mem_fun('proceed').exclude() # TODO: expose this function
z.var('rows').exclude() # TODO: expose this variable
sb.finalize_class(z)
sb.expose_class_Ptr('FilterEngine', 'cv')

# Moments
z = sb.mb.class_('Moments')
z.include()
z.decls(lambda x: 'CvMoments' in x.decl_string).exclude()

# KalmanFilter
z = sb.mb.class_('KalmanFilter')
z.include()
for t in ('predict', 'correct'):
    z.mem_fun(t).call_policies = CP.return_self()

# FeatureEvaluator
z = sb.mb.class_('FeatureEvaluator')
sb.init_class(z)
sb.finalize_class(z)
sb.expose_class_Ptr('FeatureEvaluator', 'cv')

# CascadeClassifier
z = sb.mb.class_('CascadeClassifier')
sb.init_class(z)
z.mem_fun('detectMultiScale')._transformer_creators.append(FT.arg_output('objects'))
sb.register_ti('CvHaarClassifierCascade')
sb.expose_class_Ptr('CvHaarClassifierCascade')
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
    
    std::vector<cv::Point> my_dryRun(const cv::Mat &image)
    {
        std::vector<cv::Point> pts;
        my_setImage(feval, image);
        float bias=0.0001f;
        CvHidHaarClassifierCascade* cascade = oldCascade->hid_cascade;
        int i;
        for(i = 0; i < cascade->count; ++i)
            cascade->stage_classifier[i].threshold += bias;
        cv::Point pt;
        int w1 = oldCascade->orig_window_size.width;
        int h1 = oldCascade->orig_window_size.height;
        int w = image.cols-w1;
        int h = image.rows-h1;
        double mean, var;
        for(pt.y = 0; pt.y < h; ++pt.y)
            for(pt.x = 0; pt.x < w; ++pt.x)
            {
                // mean = ((double)(sum.at<int>(pt.y, pt.x) + sum.at<int>(pt.y+h1, pt.x+w1)
                    // - sum.at<int>(pt.y+h1, pt.x) - sum.at<int>(pt.y, pt.x+w1))) / 
                    // (w1*h1);
                // var = ((double)(sqsum.at<int>(pt.y, pt.x) + sqsum.at<int>(pt.y+h1, pt.x+w1)
                    // - sqsum.at<int>(pt.y+h1, pt.x) - sqsum.at<int>(pt.y, pt.x+w1))) / 
                    // (w1*h1) - mean*mean;
                // if(var <= bias) continue;
                if(my_runAt(feval, pt) > 0) pts.push_back(pt);
            }
        for(i = 0; i < cascade->count; ++i)
            cascade->stage_classifier[i].threshold -= bias;
        return pts;
    }

''')
z.add_registration_code('def("runAt", &::CascadeClassifier_wrapper::my_runAt, ( bp::arg("_feval"), bp::arg("pt") ) )')
z.add_registration_code('def("setImage", &::CascadeClassifier_wrapper::my_setImage, ( bp::arg("_feval"), bp::arg("image") ) )')
z.add_registration_code('def("runCascade", &::CascadeClassifier_wrapper::my_dryRun, ( bp::arg("image") ) )')
sb.finalize_class(z)

# CascadeClassifier's sub-classes
for t in ('DTreeNode', 'DTree', 'Stage'):
    z2 = z.class_(t)
    sb.init_class(z2)
    sb.finalize_class(z2)
    sb.register_vec('std::vector', 'cv::CascadeClassifier::'+t, 'vector_CascadeClassifier_'+t)


# StereoBM
z = sb.mb.class_('StereoBM')
sb.init_class(z)
sb.register_ti('CvStereoBMState')
sb.expose_class_Ptr('CvStereoBMState')
sb.finalize_class(z)

# StereoSGBM
z = sb.mb.class_('StereoSGBM')
sb.init_class(z)
sb.finalize_class(z)


# KeyPoint
z = sb.mb.class_('KeyPoint')
sb.init_class(z)
sb.register_vec('std::vector', 'cv::KeyPoint')    
sb.finalize_class(z)

# SURF
z = sb.mb.class_('SURF')
sb.init_class(z)
z.operator(lambda x: len(x.arguments)==3)._transformer_creators.append(FT.arg_output('keypoints'))
z.operator(lambda x: len(x.arguments)==5)._transformer_creators.append(FT.arg_output('descriptors'))
sb.finalize_class(z)


# MSER
z = sb.mb.class_('MSER')
sb.init_class(z)
z.operator('()')._transformer_creators.append(FT.arg_output('msers'))
sb.finalize_class(z)

# StarDetector
z = sb.mb.class_('StarDetector')
sb.init_class(z)
z.operator('()')._transformer_creators.append(FT.arg_output('keypoints'))
sb.finalize_class(z)
sb.mb.class_('CvStarDetectorParams').include()

# CvLevMarq
# not yet documented, wait until requested: fix the rest of the member declarations
z = sb.mb.class_('CvLevMarq')
z.include()
z.decls().exclude()

# lsh_hash
sb.mb.class_('lsh_hash').include()

# CvLSHOperations
# not yet documented, wait until requested: fix the rest of the member declarations
z = sb.mb.class_('CvLSHOperations')
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
    sb.mb.free_funs(z).include()
    
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
for z in sb.mb.free_funs(lambda x: x.name=='write' and \
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
for z in sb.mb.free_funs(lambda x: x.name=='read' and \
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
    FT.expose_func(sb.mb.free_fun(t), return_pointee=False, 
        transformer_creators=[FT.input_array1d('src'), FT.input_array1d('dst')])
        
# goodFeaturesToTrack
FT.expose_func(sb.mb.free_fun('goodFeaturesToTrack'), return_pointee=False, 
    transformer_creators=[FT.arg_output('corners')])

# 'HoughCircles', 'HoughLines', 'HoughLinesP'
FT.expose_func(sb.mb.free_fun('HoughCircles'), return_pointee=False, 
    transformer_creators=[FT.arg_output('circles')])
FT.expose_func(sb.mb.free_fun('HoughLines'), return_pointee=False, 
    transformer_creators=[FT.arg_output('lines')])
FT.expose_func(sb.mb.free_fun('HoughLinesP'), return_pointee=False, 
    transformer_creators=[FT.arg_output('lines')])

# getOptimalNewCameraMatrix
FT.expose_func(sb.mb.free_fun('getOptimalNewCameraMatrix'), return_pointee=False, 
    transformer_creators=[FT.output_type1('validPixROI')])

# calcHist
for z in sb.mb.free_funs('calcHist'):
    FT.expose_func(z, return_pointee=False, transformer_creators=[
        FT.input_array1d('images', 'nimages'), FT.input_array1d('channels'),
        FT.input_array1d('histSize', 'dims'), FT.input_array2d('ranges')])
    z._transformer_kwds['alias'] = 'calcHist'
        
        
# calcBackProject
for z in sb.mb.free_funs('calcBackProject'):
    FT.expose_func(z, return_pointee=False, transformer_creators=[
        FT.input_array1d('images', 'nimages'), FT.input_array1d('channels'),
        FT.input_array2d('ranges')])
    z._transformer_kwds['alias'] = 'calcBackProject'
        
# floodFill
for z in sb.mb.free_funs('floodFill'):
    FT.expose_func(z, return_pointee=False, transformer_creators=[FT.output_type1('rect')])
    z._transformer_kwds['alias'] = 'floodFill'

# HuMoments
FT.expose_func(sb.mb.free_fun('HuMoments'), return_pointee=False,
    transformer_creators=[FT.output_static_array('hu', 7)])
    
# findContours
z = sb.mb.free_fun(lambda x: x.name=='findContours' and len(x.arguments)==6)
z.include()
z._transformer_kwds['alias'] = 'findContours'
z._transformer_creators.append(FT.arg_output('contours'))
z._transformer_creators.append(FT.arg_output('hierarchy'))
    
# findContours
z = sb.mb.free_fun(lambda x: x.name=='findContours' and len(x.arguments)==5)
z.include()
z._transformer_kwds['alias'] = 'findContours'
z._transformer_creators.append(FT.arg_output('contours'))

# groupRectangles
for z in sb.mb.free_funs('groupRectangles'):
    z.include()
    z._transformer_kwds['alias'] = 'groupRectangles'
    
# approxPolyDP    
for z in sb.mb.free_funs('approxPolyDP'):
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
for z in sb.mb.free_funs('convexHull'):
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
sb.mb.free_funs('undistortPoints').include()
z = sb.mb.free_fun(lambda x: x.name=='undistortPoints' and 'vector' in x.decl_string)
z._transformer_kwds['alias'] = 'undistortPoints2'
z._transformer_creators.append(FT.arg_output('dst'))
    
# findHomography
z = sb.mb.free_fun(lambda x: x.name=='findHomography' and len(x.arguments)==4).include()
z = sb.mb.free_fun(lambda x: x.name=='findHomography' and 'vector' in x.decl_string)
z.include()
z._transformer_kwds['alias'] = 'findHomography2'
z._transformer_creators.append(FT.arg_output('mask'))
    
# projectPoints
for z in sb.mb.free_funs('projectPoints'):
    z.include()
    if len(z.arguments) < 10:
        z._transformer_kwds['alias'] = 'projectPoints' 
    else:
        z._transformer_kwds['alias'] = 'projectPoints2'
    z._transformer_creators.append(FT.arg_output('imagePoints'))

# findChessboardCorners
FT.expose_func(sb.mb.free_fun('findChessboardCorners'), return_pointee=False,
    transformer_creators=[FT.arg_output('corners')])

# calibrateCamera
FT.expose_func(sb.mb.free_fun('calibrateCamera'), return_pointee=False,
    transformer_creators=[FT.arg_output('rvecs'), FT.arg_output('tvecs')])
    
# stereoRectify
for z in sb.mb.free_funs('stereoRectify'):
    z.include()
    if len(z.arguments)>15:
        z._transformer_kwds['alias'] = 'stereoRectify2'
        z._transformer_creators.extend([FT.output_type1('validPixROI1'), 
            FT.output_type1('validPixROI2')])
    else:
        z._transformer_kwds['alias'] = 'stereoRectify'

# convertPointsHomogeneous
for z in sb.mb.free_funs('convertPointsHomogeneous'):
    z.include()        
    z._transformer_kwds['alias'] = 'convertPointsHomogeneous3D' if \
        'Point3' in z.decl_string else 'convertPointsHomogeneous2D'
    z._transformer_creators.append(FT.arg_output('dst'))
    
# findFundamentalMat
for z in sb.mb.free_funs('findFundamentalMat'):
    z.include()
    if 'vector' in z.decl_string:
        z._transformer_creators.append(FT.arg_output('mask'))
        z._transformer_kwds['alias'] = 'findFundamentalMat2'

# computeCorrespondEpilines
FT.expose_func(sb.mb.free_fun('computeCorrespondEpilines'), return_pointee=False,
    transformer_creators=[FT.arg_output('lines')])



    
sb.register_ti('cv::Point_', ['int'], '_')
sb.register_ti('cv::Point_', ['float'], '_')
z = sb.register_ti('cv::Rect_', ['int'], '_')
sb.register_vec('std::vector', z, '_')
sb.register_ti('cv::Size_', ['int'], '_')
sb.register_ti('cv::TermCriteria')
sb.register_ti('cv::RotatedRect')

dtype_dict = {
    'b': 'unsigned char',
    's': 'short',
    'w': 'unsigned short',
    'i': 'int',
    'f': 'float',
    'd': 'double',
}

Vec_dict = {
    2: 'bswifd',
    3: 'bswifd',
    4: 'bswifd',
    6: 'fd',
}

Point_dict = 'ifd'

# Vec et al
for i in Vec_dict.keys():
    for suffix in Vec_dict[i]:
        z = sb.register_ti('cv::Vec', [dtype_dict[suffix], i], 'Vec%d%s' % (i, suffix))
        sb.register_vec('std::vector', z, '_')

# Point et al
for suffix in Point_dict:
    alias = 'Point2%s' % suffix
    z = sb.register_ti('cv::Point_', [dtype_dict[suffix]], alias)
    sb.register_vec('std::vector', z, '_')
    sb.register_vec('std::vector', 'std::vector< %s >' % z, '_')

# Point3 et al
for suffix in Point_dict:
    alias = 'Point3%s' % suffix
    z = sb.register_ti('cv::Point3_', [dtype_dict[suffix]], alias)
    sb.register_vec('std::vector', z, '_')
    sb.register_vec('std::vector', 'std::vector< %s >' % z, '_')




sb.done()
