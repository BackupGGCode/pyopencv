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
# cv.h
#=============================================================================


    ''')


    # Image Processing
    cc.write('''
#-----------------------------------------------------------------------------
# Image Processing
#-----------------------------------------------------------------------------

    
CV_BLUR_NO_SCALE = 0
CV_BLUR = 1
CV_GAUSSIAN = 2
CV_MEDIAN = 3
CV_BILATERAL = 4

CV_INPAINT_NS = 0
CV_INPAINT_TELEA = 1

CV_SCHARR = -1
CV_MAX_SOBEL_KSIZE = 7

CV_BGR2BGRA =   0
CV_RGB2RGBA =   CV_BGR2BGRA

CV_BGRA2BGR =   1
CV_RGBA2RGB =   CV_BGRA2BGR

CV_BGR2RGBA =   2
CV_RGB2BGRA =   CV_BGR2RGBA

CV_RGBA2BGR =   3
CV_BGRA2RGB =   CV_RGBA2BGR

CV_BGR2RGB  =   4
CV_RGB2BGR  =   CV_BGR2RGB

CV_BGRA2RGBA =  5
CV_RGBA2BGRA =  CV_BGRA2RGBA

CV_BGR2GRAY =   6
CV_RGB2GRAY =   7
CV_GRAY2BGR =   8
CV_GRAY2RGB =   CV_GRAY2BGR
CV_GRAY2BGRA =  9
CV_GRAY2RGBA =  CV_GRAY2BGRA
CV_BGRA2GRAY =  10
CV_RGBA2GRAY =  11

CV_BGR2BGR565 = 12
CV_RGB2BGR565 = 13
CV_BGR5652BGR = 14
CV_BGR5652RGB = 15
CV_BGRA2BGR565 = 16
CV_RGBA2BGR565 = 17
CV_BGR5652BGRA = 18
CV_BGR5652RGBA = 19

CV_GRAY2BGR565 = 20
CV_BGR5652GRAY = 21

CV_BGR2BGR555  = 22
CV_RGB2BGR555  = 23
CV_BGR5552BGR  = 24
CV_BGR5552RGB  = 25
CV_BGRA2BGR555 = 26
CV_RGBA2BGR555 = 27
CV_BGR5552BGRA = 28
CV_BGR5552RGBA = 29

CV_GRAY2BGR555 = 30
CV_BGR5552GRAY = 31

CV_BGR2XYZ =    32
CV_RGB2XYZ =    33
CV_XYZ2BGR =    34
CV_XYZ2RGB =    35

CV_BGR2YCrCb =  36
CV_RGB2YCrCb =  37
CV_YCrCb2BGR =  38
CV_YCrCb2RGB =  39

CV_BGR2HSV =    40
CV_RGB2HSV =    41

CV_BGR2Lab =    44
CV_RGB2Lab =    45

CV_BayerBG2BGR = 46
CV_BayerGB2BGR = 47
CV_BayerRG2BGR = 48
CV_BayerGR2BGR = 49

CV_BayerBG2RGB = CV_BayerRG2BGR
CV_BayerGB2RGB = CV_BayerGR2BGR
CV_BayerRG2RGB = CV_BayerBG2BGR
CV_BayerGR2RGB = CV_BayerGB2BGR

CV_BGR2Luv =    50
CV_RGB2Luv =    51
CV_BGR2HLS =    52
CV_RGB2HLS =    53

CV_HSV2BGR =    54
CV_HSV2RGB =    55

CV_Lab2BGR =    56
CV_Lab2RGB =    57
CV_Luv2BGR =    58
CV_Luv2RGB =    59
CV_HLS2BGR =    60
CV_HLS2RGB =    61

CV_COLORCVT_MAX = 100

CV_INTER_NN     = 0
CV_INTER_LINEAR = 1
CV_INTER_CUBIC  = 2
CV_INTER_AREA = 3

CV_WARP_FILL_OUTLIERS = 8
CV_WARP_INVERSE_MAP = 16

CV_SHAPE_RECT = 0
CV_SHAPE_CROSS = 1
CV_SHAPE_ELLIPSE = 2
CV_SHAPE_CUSTOM = 100

CV_MOP_OPEN = 2
CV_MOP_CLOSE = 3
CV_MOP_GRADIENT = 4
CV_MOP_TOPHAT = 5
CV_MOP_BLACKHAT = 6



    ''')

    # pointers which are not Cv... * are excluded until further requested
    for z in (
        'CvFeatureTree', 'CvLSH', 'CvLSHOperations',
        'CvSURFPoint', 'CvSURFParams',
        'CvMSERParams', 
        'CvStarKeypoint',
        'CvStarDetectorParams', 
        'CvPOSITObject',
        'CvStereoBMState', 'CvStereoGCState', 
        ):
        k = mb.class_(z)
        k.include()
        try:
            vv = k.vars()
        except RuntimeError:
            vv = []
        for v in vv:
            if D.is_pointer(v.type):
                if 'Cv' in v.type.decl_string:
                    FT.expose_member_as_pointee(k, v.name)
                else:
                    v.exclude()

    # functions
    # for z in (
        # 'cvCopyMakeBorder', 'cvSmooth', 'cvFilter2D', 'cvIntegral', 'cvPyrDown', 'cvPyrUp',
        # 'cvPyrMeanShiftFiltering', 'cvWatershed', 'cvInpaint', 'cvSobel', 'cvLaplace',
        # 'cvCvtColor', 'cvResize', 'cvWarpAffine', 'cvWarpPerspective',
        # 'cvRemap', 'cvConvertMaps', 'cvLogPolar', 'cvLinearPolar',
        # 'cvErode', 'cvDilate', 'cvMorphologyEx', 
        # 'cvMoments', 'cvGetSpatialMoment', 'cvGetCentralMoment', 'cvGetNormalizedCentralMoment', 'cvGetHuMoments',
        
        # ):
        # mb.free_fun(z).include()
        
    # TODO: fix these functions:
    # cvCreatePyramid, cvReleasePyramid, cvPyrSegmentation
    # cvCreateStructuringElementEx, cvReleaseStructuringElement

    # TODO: fix this function cvPointSeqFromMat()
    mb.free_fun('cvPointSeqFromMat').exclude()

    # cvGetAffineTransform
    # FT.expose_func(mb.free_fun('cvGetAffineTransform'), return_arg_index=3)

    # cv2DRotationMatrix
    # FT.expose_func(mb.free_fun('cv2DRotationMatrix'), return_arg_index=4)
    
    # cvGetPerspectiveTransform
    # FT.expose_func(mb.free_fun('cvGetPerspectiveTransform'), return_arg_index=3)

    
