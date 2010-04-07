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
# cvaux.h
#=============================================================================


    ''')

    FT.expose_func(mb.free_fun('cvSegmentImage'), ward_indices=(5,))

    # Eigen Objects -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Eigen Objects
#-----------------------------------------------------------------------------

    
    ''')

    # 1D/2D HMM - TODO
    cc.write('''
#-----------------------------------------------------------------------------
# 1D/2D HMM
#-----------------------------------------------------------------------------

    
    ''')

    # A few functions from old stereo gesture recognition demosions - TODO
    cc.write('''
#-----------------------------------------------------------------------------
# A few functions from old stereo gesture recognition demosions
#-----------------------------------------------------------------------------

    
    ''')

    # Additional operations on Subdivisions -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Additional operations on Subdivisions
#-----------------------------------------------------------------------------

    
    ''')

    # More operations on sequences -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# More operations on sequences
#-----------------------------------------------------------------------------

    
    ''')

    # subsections:
    # Stereo correspondence -- TODO
    cc.write('''
CV_UNDEF_SC_PARAM = 12345

CV_IDP_BIRCHFIELD_PARAM1  = 25    
CV_IDP_BIRCHFIELD_PARAM2  = 5
CV_IDP_BIRCHFIELD_PARAM3  = 12
CV_IDP_BIRCHFIELD_PARAM4  = 15
CV_IDP_BIRCHFIELD_PARAM5  = 25

CV_DISPARITY_BIRCHFIELD  = 0    


    ''')
    
    mb.free_fun('cvFindStereoCorrespondence').include()
    
    # Epiline functions -- TODO

    # Contour Morphing -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Contour Morphing
#-----------------------------------------------------------------------------

    
    ''')

    # functions -- not in cvaux200.dll.a!!!
    # FT.expose_func(mb.free_fun('cvCalcContoursCorrespondence'), ward_indices=(3,))
    # FT.expose_func(mb.free_fun('cvMorphContours'), ward_indices=(5,))


    # Texture Descriptors -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Texture Descriptors
#-----------------------------------------------------------------------------

    
    ''')

    # Face eyes&mouth tracking -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Face eyes&mouth tracking
#-----------------------------------------------------------------------------

    
    ''')

    # 3D Tracker -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# 3D Tracker
#-----------------------------------------------------------------------------

    
    ''')

    # Skeletons and Linear-Contour Models -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Skeletons and Linear-Contour Models
#-----------------------------------------------------------------------------

    
    ''')

    # Background/foreground segmentation -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Background/foreground segmentation
#-----------------------------------------------------------------------------

    
    ''')

    # Calibration engine -- TODO
    cc.write('''
#-----------------------------------------------------------------------------
# Calibration engine
#-----------------------------------------------------------------------------

    
    ''')
