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
sb.load_regs('cxcore_hpp_reg.sdd')



sb.cc.write('''
import numpy as _np
    
#=============================================================================
# sdopencv
#=============================================================================

def cmpsum(arr, thresh=0, pos_val=1, neg_val=None):
    """Compares and sums up.
    
    Description:
        return np.sum(np.where(arr >= thresh, pos_val, neg_val), axis=-1)
        
    Input:
        arr : ndarray
            input array
        thresh : 1D ndarray or a number (default value: 0)
        pos_val : 1D ndarray or a number (default value: 1)
        neg_val : 1D ndarray or a number (default value: -pos_val)
    """
    if neg_val is None:
        neg_val = -pos_val
    return _np.sum(_np.where(arr >= thresh, pos_val, neg_val), axis=-1)

''')

sdopencv = sb.mb.namespace('sdopencv')
sdopencv.include()

for t in ('DifferentialImage', 'IntegralImage', 'IntegralHistogram'):
    z = sdopencv.class_(t)
    sb.init_class(z)
    sb.finalize_class(z)
    
for t in ('LUTFunc', 'StumpFunc'):
    z = sdopencv.class_(t)
    sb.asClass(z, sdopencv.class_('StepFunc'))

sb.done()
sb.save_regs('sdopencv_reg.sdd')
