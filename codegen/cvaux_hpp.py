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
# cvaux.hpp
#=============================================================================


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # CvCamShiftTracker
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvCamShiftTracker')
    z.include()
    z.decls().exclude()
    
    # CvAdaptiveSkinDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvAdaptiveSkinDetector')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyPoint
    mb.class_('CvFuzzyPoint').include()
    
    # CvFuzzyCurve
    mb.class_('CvFuzzyCurve').include()
    
    # CvFuzzyFunction
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyFunction')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyRule
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyRule')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyController
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyController')
    z.include()
    z.decls().exclude()
    
    # CvFuzzyMeanShiftTracker
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvFuzzyMeanShiftTracker')
    z.include()
    z.decls().exclude()
    
    # Octree
    # TODO: fix the rest of the member declarations
    z = mb.class_('Octree')
    z.include()
    z.decls().exclude()
    
    # Mesh3D
    # TODO: fix the rest of the member declarations
    z = mb.class_('Mesh3D')
    z.include()
    z.decls().exclude()
    
    # SpinImageModel
    # TODO: fix the rest of the member declarations
    z = mb.class_('SpinImageModel')
    z.include()
    z.decls().exclude()
    
    # TickMeter
    mb.class_('TickMeter').include()
    
    # HOGDescriptor
    # TODO: fix the rest of the member declarations
    z = mb.class_('HOGDescriptor')
    z.include()
    z.decls().exclude()
    
    # SelfSimDescriptor
    # TODO: fix the rest of the member declarations
    z = mb.class_('SelfSimDescriptor')
    z.include()
    z.decls().exclude()
    
    # PatchGenerator
    mb.class_('PatchGenerator').include()
    
    # LDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('LDetector')
    z.include()
    z.decls().exclude()
    
    # FernClassifier
    # TODO: fix the rest of the member declarations
    z = mb.class_('FernClassifier')
    z.include()
    z.decls().exclude()
    
    # PlanarObjectDetector
    # TODO: fix the rest of the member declarations
    z = mb.class_('PlanarObjectDetector')
    z.include()
    z.decls().exclude()
    
    # OneWayDescriptor
    # TODO: fix the rest of the member declarations
    z = mb.class_('OneWayDescriptor')
    z.include()
    z.decls().exclude()
    
    # OneWayDescriptorBase
    # TODO: fix the rest of the member declarations
    z = mb.class_('OneWayDescriptorBase')
    z.include()
    z.decls().exclude()
    
    # OneWayDescriptorObject
    # TODO: fix the rest of the member declarations
    z = mb.class_('OneWayDescriptorObject')
    z.include()
    z.decls().exclude()
    
    # LevMarqSparse
    # TODO: fix the rest of the member declarations
    z = mb.class_('LevMarqSparse')
    z.include()
    z.decls().exclude()

    
    #=============================================================================
    # Free Functions
    #=============================================================================

    # TODO:
    # TickMeter's operator <<
    # findOneWayDescriptor
    # FAST
