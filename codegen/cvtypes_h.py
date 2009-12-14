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

import cxtypes_h

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cvtypes.h
#=============================================================================


# contour retrieval mode
CV_RETR_EXTERNAL = 0
CV_RETR_LIST     = 1
CV_RETR_CCOMP    = 2
CV_RETR_TREE     = 3

# contour approximation method
CV_CHAIN_CODE               = 0
CV_CHAIN_APPROX_NONE        = 1
CV_CHAIN_APPROX_SIMPLE      = 2
CV_CHAIN_APPROX_TC89_L1     = 3
CV_CHAIN_APPROX_TC89_KCOS   = 4
CV_LINK_RUNS                = 5

# Haar-like Object Detection structures

CV_HAAR_MAGIC_VAL    = 0x42500000
CV_TYPE_NAME_HAAR    = "opencv-haar-classifier"
CV_HAAR_FEATURE_MAX  = 3


    ''')


    z = mb.class_('CvConnectedComp')
    z.include()
    FT.expose_member_as_pointee(z, 'contour')

    # CvContourScanner
    z = mb.class_('_CvContourScanner')
    z.include()
    z.rename('CvContourScanner')
    mb.insert_del_interface('CvContourScanner', '_PE._cvEndFindContours')

    # CvChainPtReader
    z = mb.class_('CvChainPtReader')
    z.include()
    cxtypes_h.expose_CvSeqReader_members(z, FT)
    z.var('deltas').exclude() # wait until requested

    # CvContourTree
    z = mb.class_('CvContourTree')
    z.include()
    cxtypes_h.expose_CvSeq_members(z, FT)

    #CvConvexityDefect
    z = mb.class_('CvConvexityDefect')
    z.include()
    for t in (
        'start', 'end', 'depth_point',
        ):
        FT.expose_member_as_pointee(z, t)


    def expose_QuadEdge2D_members(z):
        FT.expose_member_as_array_of_pointees(z, 'pt', 4)
        
    z = mb.class_('CvQuadEdge2D')
    z.include()
    expose_QuadEdge2D_members(z)

    mb.class_('CvSubdiv2DPoint').include()
    mb.decl('CvSubdiv2DEdge').include()

    # CvSubdiv2D
    z = mb.class_('CvSubdiv2D')
    z.include()
    cxtypes_h.expose_CvGraph_members(z, FT)
    
    
    for z in (
        'CvVect32f', 'CvMatr32f', 'CvVect64d', 'CvMatr64d',
        ):
        mb.decl(z).include()
        
    # pointers which are not Cv... * are excluded until further requested
    for z in (
        'CvAvgComp',
        'CvHaarClassifier', 'CvHaarStageClassifier', 'CvHaarClassifierCascade',
        ):
        k = mb.class_(z)
        k.include()
        for v in k.vars():
            if D.is_pointer(v.type):
                if 'Cv' in v.type.decl_string:
                    FT.expose_member_as_pointee(k, v.name)
                else:
                    v.exclude()

    # CvHaarFeature
    z = mb.class_('CvHaarFeature')
    z.include()
    for t in ('r', 'weight', 'rect'): # wait until requested: expose the member variables
        z.decl(t).exclude()

    
                    
    # CvConDensation
    z = mb.class_('CvConDensation')
    z.include()
    for arg in z.vars():
        if D.is_pointer(arg.type):
            arg.exclude() # wait until requested
    mb.insert_del_interface('CvConDensation', '_PE._cvReleaseConDensation')

