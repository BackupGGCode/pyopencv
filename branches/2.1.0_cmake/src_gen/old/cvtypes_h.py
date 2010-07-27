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
import cxtypes_h

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cvtypes.h
#=============================================================================


# Defines for Distance Transform
CV_DIST_USER    = -1
CV_DIST_L1      = 1
CV_DIST_L2      = 2
CV_DIST_C       = 3
CV_DIST_L12     = 4
CV_DIST_FAIR    = 5
CV_DIST_WELSCH  = 6
CV_DIST_HUBER   = 7

# Haar-like Object Detection structures

CV_HAAR_MAGIC_VAL    = 0x42500000
CV_TYPE_NAME_HAAR    = "opencv-haar-classifier"
CV_HAAR_FEATURE_MAX  = 3

def CV_IS_HAAR_CLASSIFIER(haar_cascade):
    return isinstance(haar_cascade, CvHaarClassifierCascade) and \
        haar_cascade.flags&CV_MAGIC_MASK==CV_HAAR_MAGIC_VAL

    ''')


    z = mb.class_('CvConnectedComp')
    mb.init_class(z)
    common.register_vec('std::vector', 'CvConnectedComp', 'vector_CvConnectedComp')
    mb.finalize_class(z)
    mb.expose_class_Seq('CvConnectedComp')

    # CvContourScanner
    z = mb.class_('_CvContourScanner')
    mb.init_class(z)
    z.rename('CvContourScanner')
    mb.insert_del_interface('CvContourScanner', '_PE._cvEndFindContours')
    mb.finalize_class(z)

    # CvChainPtReader
    z = mb.class_('CvChainPtReader')
    mb.init_class(z)
    cxtypes_h.expose_CvSeqReader_members(z, FT)
    z.var('deltas').exclude() # wait until requested
    mb.finalize_class(z)

    # CvContourTree
    z = mb.class_('CvContourTree')
    mb.init_class(z)
    cxtypes_h.expose_CvSeq_members(z, FT)
    mb.finalize_class(z)

    #CvConvexityDefect
    z = mb.class_('CvConvexityDefect')
    mb.init_class(z)
    for t in (
        'start', 'end', 'depth_point',
        ):
        FT.expose_member_as_pointee(z, t)
    mb.finalize_class(z)

    def expose_QuadEdge2D_members(z):
        FT.expose_member_as_array_of_pointees(z, 'pt', 4)
        FT.set_array_item_type_as_size_t(z, 'next')
        
    z = mb.class_('CvQuadEdge2D')
    mb.init_class(z)
    expose_QuadEdge2D_members(z)
    mb.finalize_class(z)

    # CvSubdiv2DPoint
    z = mb.class_('CvSubdiv2DPoint')
    mb.init_class(z)
    mb.decl('CvSubdiv2DEdge').include()
    mb.finalize_class(z)

    # CvSubdiv2D
    z = mb.class_('CvSubdiv2D')
    mb.init_class(z)
    cxtypes_h.expose_CvGraph_members(z, FT)
    mb.finalize_class(z)
    
    
    for z in (
        'CvVect32f', 'CvMatr32f', 'CvVect64d', 'CvMatr64d',
        ):
        mb.decl(z).include()
        
    # pointers which are not Cv... * are excluded until further requested
    for z in (
        'CvAvgComp',
        ):
        k = mb.class_(z)
        mb.init_class(k)
        for v in k.vars():
            if D.is_pointer(v.type):
                if 'Cv' in v.type.decl_string:
                    FT.expose_member_as_pointee(k, v.name)
                else:
                    v.exclude()
        mb.finalize_class(k)

    # CvHaarFeature
    z = mb.class_('CvHaarFeature')
    mb.init_class(z)
    for t in ('r', 'weight', 'rect'):
        z.decl(t).exclude()
    z.add_declaration_code('''
static cv::Rect *get_rect(CvHaarFeature const &inst, int i)
{
    return (cv::Rect *)&inst.rect[i].r;
}

static float get_rect_weight(CvHaarFeature const &inst, int i)
{
    return inst.rect[i].weight;
}

static void set_rect_weight(CvHaarFeature &inst, int i, float _weight)
{
    inst.rect[i].weight = _weight;
}

    ''')
    z.add_registration_code('def("get_rect", &::get_rect, bp::return_internal_reference<>())')
    z.add_registration_code('def("get_rect_weight", &::get_rect_weight)')
    z.add_registration_code('def("set_rect_weight", &::set_rect_weight)')
    mb.finalize_class(z)
    
    # CvHaarClassifier
    z = mb.class_('CvHaarClassifier')
    mb.init_class(z)
    FT.expose_member_as_pointee(z, 'haar_feature')
    FT.expose_array_member_as_Mat(z, 'threshold', 'count')
    FT.expose_array_member_as_Mat(z, 'left', 'count')
    FT.expose_array_member_as_Mat(z, 'right', 'count')
    FT.expose_array_member_as_Mat(z, 'alpha', 'count', '1')
    mb.finalize_class(z)
    
    # CvHaarStageClassifier
    z = mb.class_('CvHaarStageClassifier')
    mb.init_class(z)
    FT.expose_member_as_array_of_pointees(z, 'classifier', 'inst.count')
    mb.finalize_class(z)
    
    # CvHaarClassifierCascade
    z = mb.class_('CvHaarClassifierCascade')
    mb.init_class(z)
    FT.expose_member_as_array_of_pointees(z, 'stage_classifier', 'inst.count')
    FT.expose_member_as_pointee(z, 'hid_cascade')
    mb.finalize_class(z)
    
    # CvHidHaarFeature
    z = mb.class_('CvHidHaarFeature')
    mb.init_class(z)
    for t in ('p0', 'p1', 'p2', 'p3', 'weight', 'rect'):
        z.decl(t).exclude()
    z.add_declaration_code('''
static float get_rect_weight(CvHidHaarFeature const &inst, int i)
{
    return inst.rect[i].weight;
}

static void set_rect_weight(CvHidHaarFeature &inst, int i, float _weight)
{
    inst.rect[i].weight = _weight;
}

    ''')
    z.add_registration_code('def("get_rect_weight", &::get_rect_weight)')
    z.add_registration_code('def("set_rect_weight", &::set_rect_weight)')
    mb.finalize_class(z)
    
    # CvHidHaarTreeNode
    z = mb.class_('CvHidHaarTreeNode')
    mb.init_class(z)
    mb.finalize_class(z)
    
    # CvHidHaarClassifier
    z = mb.class_('CvHidHaarClassifier')
    mb.init_class(z)
    FT.expose_member_as_array_of_pointees(z, 'node', 'inst.count')
    FT.expose_array_member_as_Mat(z, 'alpha', 'count', '1')
    mb.finalize_class(z)
    
    # CvHidHaarStageClassifier
    z = mb.class_('CvHidHaarStageClassifier')
    mb.init_class(z)
    FT.expose_member_as_array_of_pointees(z, 'classifier', 'inst.count')
    FT.expose_member_as_pointee(z, 'next')
    FT.expose_member_as_pointee(z, 'parent')
    FT.expose_member_as_pointee(z, 'child')
    mb.finalize_class(z)
    
    # CvHidHaarClassifierCascade
    z = mb.class_('CvHidHaarClassifierCascade')
    mb.init_class(z)
    for t in ('pq0', 'pq1', 'pq2', 'pq3', 'p0', 'p1', 'p2', 'p3', 'ipp_stages'):
        z.var(t).exclude()
    FT.expose_member_as_array_of_pointees(z, 'stage_classifier', 'inst.count')
    mb.finalize_class(z)
    
                    
