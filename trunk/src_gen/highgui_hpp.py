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
# highgui.hpp
#=============================================================================


    ''')


    # Basic GUI functions 
    cc.write('''
#-----------------------------------------------------------------------------
# C++ Interface
#-----------------------------------------------------------------------------

    ''')

    # functions
    for z in (
        'namedWindow', 'imshow', 'imread', 'imwrite', 'imencode', 'imdecode', 'waitKey',
        'setWindowProperty', 'getWindowProperty',
        ):
        mb.free_fun(z).include()
        
    mb.free_fun('imencode')._transformer_creators.append(FT.arg_std_vector('buf', 2))
        
    # createTrackbar
    z = mb.free_fun('createTrackbar')
    FT.expose_func(z, return_pointee=False, transformer_creators=[
        FT.trackbar_callback2_func('onChange', 'userdata'), FT.from_address('value')])
    FT.add_underscore(z)
    cc.write('''
def createTrackbar(trackbar_name, window_name, value, count, on_change=None, userdata=None):
    if not isinstance(value, _CT.c_int):
        value = _CT.c_int(value)

    result, z = _PE._createTrackbar(trackbar_name, window_name, _CT.addressof(value), count, on_change, userdata=userdata)
    if result:
        cb_key = 'tracker-' + trackbar_name
        _windows_callbacks.setdefault(window_name,{})[cb_key] = z
    return result
createTrackbar.__doc__ = _PE._createTrackbar.__doc__
    ''')
    mb.add_doc('createTrackbar', "'value' is the initial position of the trackbar. Also, if 'value' is an instance of ctypes.c_int, it keeps the current position of the trackbar at any time.", "'onChange' can be passed with None.")

    # VideoCapture
    z = mb.class_('VideoCapture')
    mb.init_class(z)
    z.operator('>>').exclude()
    z.add_declaration_code('static cv::VideoCapture &rshift( cv::VideoCapture &inst, cv::Mat &x ){ return inst >> x; }')
    z.add_registration_code('def( "__rshift__", &::rshift, bp::return_self<>() )')
    mb.finalize_class(z)
    
    # VideoWriter
    z = mb.class_('VideoWriter')
    mb.init_class(z)
    z.operator('<<').exclude()
    z.add_declaration_code('static cv::VideoWriter &lshift( cv::VideoWriter &inst, cv::Mat const &x ){ return inst << x; }')
    z.add_registration_code('def( "__lshift__", &::lshift, bp::return_self<>() )')
    mb.finalize_class(z)
            

