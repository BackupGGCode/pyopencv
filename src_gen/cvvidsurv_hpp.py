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
# cvvidsurf.hpp
#=============================================================================

CV_BLOB_MINW = 5
CV_BLOB_MINH = 5


    ''')

    #=============================================================================
    # Structures
    #=============================================================================

    # CvDefParam
    # TODO: fix the rest of the member declarations
    z = mb.class_('CvDefParam')
    z.include()
    FT.expose_member_as_pointee(z, 'next')
    for t in ('pName', 'pComment', 'Str'):
        FT.expose_member_as_str(z, t)
    for t in ('pDouble', 'pFloat', 'pInt', 'pStr'):
        z.var(t).exclude()
    
    # CvVSModule
    z = mb.class_('CvVSModule')
    mb.init_class(z)
    z.mem_fun('GetModuleName').exclude()
    z.add_declaration_code('''    
inline bp::str CvVSModule_GetModuleName(CvVSModule &inst) {  return bp::str(inst.GetModuleName()); }

    ''')
    z.add_registration_code('def("GetModuleName", &::CvVSModule_GetModuleName)')
    mb.finalize_class(z)
    
    # CvFGDetector
    # TODO: fix this guy
    # z = mb.class_('CvFGDetector')
    # z.include()
    # z.decls().exclude()
    
    # CvBlob
    mb.class_('CvBlob').include()
    mb.free_fun('cvBlob').include()
    
    # CvBlobSeq
    z = mb.class_('CvBlobSeq')
    mb.init_class(z)
    mb.finalize_class(z)
    
    # CvBlobTrack
    z = mb.class_('CvBlobTrack')
    z.include()
    FT.expose_member_as_pointee(z, 'pBlobSeq')
    
    # CvBlobTrackSeq
    z = mb.class_('CvBlobTrackSeq')
    mb.init_class(z)
    mb.finalize_class(z)
    
    # CvBlobDetector
    # TODO: fix this guy
    # z = mb.class_('CvBlobDetector')
    # z.include()
    # z.decls().exclude()
    
    # CvBlob
    mb.class_('CvDetectedBlob').include()
    mb.free_fun('cvDetectedBlob').include()
    
    # CvObjectDetector
    z = mb.class_('CvObjectDetector')
    mb.init_class(z)
    mb.finalize_class(z)
    
    
    
    #=============================================================================
    # Free Functions
    #=============================================================================

    # TODO:
    # cvWriteStruct, cvReadStructByName, 
    # findOneWayDescriptor
    
