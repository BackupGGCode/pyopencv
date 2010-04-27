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
# cvcompat.h
#=============================================================================


CvPoint2D64d = CvPoint2D64f
CvPoint3D64d = CvPoint3D64f

CV_MAT32F      = CV_32FC1
CV_MAT3x1_32F  = CV_32FC1
CV_MAT4x1_32F  = CV_32FC1
CV_MAT3x3_32F  = CV_32FC1
CV_MAT4x4_32F  = CV_32FC1

CV_MAT64D      = CV_64FC1
CV_MAT3x1_64D  = CV_64FC1
CV_MAT4x1_64D  = CV_64FC1
CV_MAT3x3_64D  = CV_64FC1
CV_MAT4x4_64D  = CV_64FC1

IPL_GAUSSIAN_5x5   = 7
CvBox2D32f     = CvBox2D

# TODO: fix these functions
# cvIntegralImage     = cvIntegral
# cvMatchContours     = cvMatchShapes

cvCvtPixToPlane = cvSplit
cvCvtPlaneToPix = cvMerge

cvPseudoInv = cvPseudoInverse

    ''')

    for z in (
        'cvMean', 'cvSumPixels', 'cvMean_StdDev',
        'cvmPerspectiveProject', 'cvFillImage',
        'cvRandSetRange', 'cvRandInit', 'cvRand',
        'cvContourBoundingRect', 'cvPseudoInverse',
        ):
        mb.free_fun(z).include()
        
    mb.class_('CvRandState').include()

    # TODO: fix these functions
    # cvRandNext, cvb*

    # TODO: check the rest of the functions in cvcompat.h and try to wrap them, if necessary

