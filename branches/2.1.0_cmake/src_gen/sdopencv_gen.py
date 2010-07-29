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
sb = sdpypp.SdModuleBuilder('sdopencv')



    
sb.register_vec('std::vector', 'unsigned char', excluded=True)
sb.register_vec('std::vector', 'int', excluded=True)
sb.register_vec('std::vector', 'unsigned int', excluded=True)
sb.register_vec('std::vector', 'float', excluded=True)
sb.register_ti('cv::Mat')
sb.register_vec('std::vector', 'cv::Mat', excluded=True)
sb.register_ti('cv::MatND')
sb.register_vec('std::vector', 'cv::MatND', excluded=True)
sb.register_ti('cv::Ptr', ['cv::Mat'])
sb.register_vec('std::vector', 'cv::Ptr<cv::Mat>', excluded=True)
sb.register_ti('cv::KeyPoint')
sb.register_vec('std::vector', 'cv::KeyPoint', excluded=True)
sb.register_ti('cv::Scalar_', ['double'], 'Scalar')
sb.register_vec('std::vector', 'cv::Scalar_<double>', excluded=True)

z = sb.register_ti('cv::Rect_', ['int'], 'Rect')
sb.register_vec('std::vector', z, excluded=True)
sb.register_ti('cv::Size_', ['int'], 'Size')
sb.register_ti('cv::TermCriteria')
sb.register_ti('cv::RotatedRect')
sb.register_ti('cv::Range')

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
        sb.register_vec('std::vector', z, excluded=True)

# Point et al
for suffix in Point_dict:
    alias = 'Point2%s' % suffix
    z = sb.register_ti('cv::Point_', [dtype_dict[suffix]], alias)
    sb.register_vec('std::vector', z, excluded=True)
    sb.register_vec('std::vector', 'std::vector< %s >' % z, excluded=True)

# Point3 et al
for suffix in Point_dict:
    alias = 'Point3%s' % suffix
    z = sb.register_ti('cv::Point3_', [dtype_dict[suffix]], alias)
    sb.register_vec('std::vector', z, excluded=True)
    sb.register_vec('std::vector', 'std::vector< %s >' % z, excluded=True)


    
    
sb.cc.write('''
#=============================================================================
# sdopencv
#=============================================================================


''')

sdopencv = sb.mb.namespace('sdopencv')
sdopencv.include()

for t in ('DifferentialImage', 'IntegralImage', 'IntegralHistogram'):
    z = sdopencv.class_(t)
    sb.init_class(z)
    sb.finalize_class(z)

sb.done()

