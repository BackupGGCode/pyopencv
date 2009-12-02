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

    # TODO: Base...Filter
    # for t in ('BaseRowFilter', 'BaseColumnFilter', 'BaseFilter'):
        # z = mb.class_(t)
        # z.include()
        # z.operator('()').exclude() # TODO: fix this function
    
    # FilterEngine
    # TODO: fix the rest of the member declarations
    z = mb.class_('FilterEngine')
    z.include()
    z.decls().exclude()
    
    # Moments
    z = mb.class_('Moments')
    z.include()
    z.operator(lambda x: x.name.endswith('::CvMoments')).rename('as_CvMoments')
    
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
    # TODO: fix the rest of the member declarations
    z = mb.class_('StereoBM')
    z.include()
    z.decls().exclude()
    
    # KeyPoint
    mb.class_('KeyPoint').include()
    
    # SURF
    # TODO: fix the rest of the member declarations
    z = mb.class_('SURF')
    z.include()
    z.decls().exclude()
    
    # MSER
    # TODO: fix the rest of the member declarations
    z = mb.class_('MSER')
    z.include()
    z.decls().exclude()
    
    # StarDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('StarDetector')
    z.include()
    z.decls().exclude()
    
    # CvLevMarq
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvLevMarq')
    z.include()
    z.decls().exclude()
    
    # lsh_hash
    mb.class_('lsh_hash').include()
    
    # CvLSHOperations
    # TODO: fix the rest of the member declarations
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
        'cornerEigenValsAndVecs', 'preCornerDetect', 'erode', 'dilate', 
        'morphologyEx', 'resize', 'warpAffine', 'warpPerspective', 'remap',
        'convertMaps', 'getRotationMatrix2D', 'invertAffineTransform', 
        'getRectSubPix', 'integral', 'accumulate', 'accumulateSquare', 
        'accumulateProduct', 'accumulateWeighted', 'threshold', 
        'adaptiveThreshold', 'pyrDown', 'pyrUp', 'undistort', 
        'initUndistortRectifyMap', 'getDefaultNewCameraMatrix', 
        'calcOpticalFlowFarneback', 'compareHist', 'equalizeHist', 'watershed',
        'inpaint', 'distanceTransform', 'cvtColor', 'moments', 'matchTemplate',
        'arcLength', 'boundingRect', 'contourArea', 'minAreaRect', 
        'minEnclosingCircle', 'matchShapes', 'isContourConvex', 'fitEllipse',
        'fitLine', 'pointPolygonTest', 'updateMotionHistory', 
        'calcMotionGradient', 'calcGlobalOrientation', 'CamShift', 'meanShift', 
        'Rodrigues', 'RQDecomp3x3', 'decomposeProjectionMatrix', 'matMulDeriv', 
        'composeRT', 'solvePnP', 'drawChessboardCorners', 
        'calibrationMatrixValues', 'stereoRectify', 'stereoRectifyUncalibrated', 
        'reprojectImageTo3D', 
        ):
        mb.free_funs(z).include()

    # TODO:
    # getLinearRowFilter, getLinearColumnFilter, getLinearFilter, createSeparableLinearFilter, createLinearFilter
    # createGaussianFilter,  createDerivFilter, getRowSumFilter, getColumnSumFilter, createBoxFilter
    # getMorphologyRowFilter, getMorphologyColumnFilter, getMorphologyFilter
    # createMorphologyFilter,  cornerSubPix, goodFeaturesToTrack, HoughLines, HoughLinesP, HoughCircles,
    # getPerspectiveTransform, getAffineTransform, buildPyramid, calcOpticalFlowPyrLK
    # calcHist, calcBackProject, floodFill, HuMoments, findContours, drawContours, approxPolyDP
    # convexHull, estimateAffine3D, groupRectangles, undistortPoints, findHomography
    # projectPoints, initCameraMatrix2D, findChessboardCorners, calibrateCamera, stereoCalibrate
    # convertPointsHomogeneous, findFundamentalMat, computeCorrespondEpilines
    # write, read
    
    # TODO: missing functions
    # 'estimateRigidTransform', 
    
