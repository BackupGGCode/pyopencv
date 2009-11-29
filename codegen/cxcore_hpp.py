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
# cxcore.hpp
#=============================================================================


    ''')

    mb.class_('Exception').include()

    for z in ('fromUtf16', 'toUtf16',
        'setNumThreads', 'getNumThreads', 'getThreadNum',
        'getTickCount', 'getTickFrequency',
        'setUseOptimized', 'useOptimized',
        ):
        mb.free_fun(lambda decl: z in decl.name).include()
        
    # TODO:
    # Vec et al
    # Complex et al
    # Point et al
    # Point3 et al
    # Size et al
    # Rect et al
    # RotatedRect
    # Scalar et al
    # Range
    # DataType et al
    # RNG
    # TermCriteria
    
    for z in (
        'getElemSize',
        # 'cvarrToMat', 'extractImageCOI', 'insertImageCOI', # removed, everything is in ndarray now
        'convertScaleAbs', 'LUT', 'reduce', 'flip',
        ):
        mb.free_funs(z).include()

    # add
    mb.free_funs('add').exclude()
    mb.add_registration_code('''bp::def("add", &sd::add,
        ( bp::arg("a"), bp::arg("b"), bp::arg("c"), bp::arg("mask")=bp::object() ));''')
    
    # subtract
    mb.free_funs('subtract').exclude()
    mb.add_registration_code('''bp::def("subtract", &sd::subtract,
        ( bp::arg("a"), bp::arg("b"), bp::arg("c"), bp::arg("mask")=bp::object() ));''')
    
    # mean
    mb.free_funs('mean').exclude()
    mb.add_registration_code('''bp::def("mean", &sd::mean,
        ( bp::arg("m"), bp::arg("mask")=bp::object() ));''')
    
    # meanStdDev
    mb.free_funs('meanStdDev').exclude()
    mb.add_registration_code('''bp::def("meanStdDev", &sd::meanStdDev,
        ( bp::arg("m"), bp::arg("mean"), bp::arg("stddev"), bp::arg("mask")=bp::object() ));''')
    
    # free functions which are rendered obsolete by numpy
    for z in (
        'multiply', 'divide', 'scaleAdd', 'addWeighted', 'sum', 'countNonZero', 
        'split', 'merge', 'mixChannels',
    ):
        mb.free_funs(z).exclude()
        
    # TODO: function norm, normalize, minMaxLoc
    
    mb.free_fun('repeat', return_type=D.dummy_type_t('void')).include()
    
    

    # TODO: expand the rest of cxcore.hpp
 
    

    # for z in ('DataDepth', 'Vec', 'Point', 'Size', 'Rect', 'RotatedRect',
        # 'Scalar', 'Range', 'DataType', 
        # 'RNG', 'TermCriteria',
        # ):
        # try:
            # mb.classes(lambda decl: decl.name.startswith(z)).include()
        # except RuntimeError:
            # pass

    # TODO: fix these things


