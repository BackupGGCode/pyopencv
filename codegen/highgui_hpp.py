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
# highgui.hpp
#=============================================================================


    ''')


    # Basic GUI functions 
    cc.write('''
#-----------------------------------------------------------------------------
# C++ Interface
#-----------------------------------------------------------------------------

createTrackbar = cvCreateTrackbar2
getTrackbarPos = cvGetTrackbarPos # don't know why they haven't exported this function
setTrackbarpos = cvSetTrackbarPos # don't know why they haven't exported this function
    
    ''')

    # functions
    for z in (
        'namedWindow', 'imshow', 'imread', 'imwrite', 'imdecode', 'waitKey',
        ):
        mb.free_fun(z).include()
        
    FT.expose_func(mb.free_fun('imencode'), return_pointee=False, transformer_creators=[FT.output_ndarray('buf')])
    # FT.expose_func(mb.free_fun('imread'), return_pointee=False, return_as_ndarray_from_type='cv::Mat')
    # z = mb.free_fun('imread')
    # z.call_policies = CP.custom_call_policies( "bp::return_value_policy<bp::return_by_value>", "opencv_extra.hpp" )
        
    # classes
    for z in (
        'VideoCapture', 'VideoWriter',
        ):
        mb.class_(z).include()
            

