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
sb = sdpypp.SdModuleBuilder('highgui')



    
sb.register_vec('std::vector', 'unsigned char', excluded=True)
sb.register_vec('std::vector', 'int', excluded=True)
sb.register_vec('std::vector', 'unsigned int', excluded=True)
sb.register_vec('std::vector', 'float', excluded=True)
sb.register_ti('cv::Mat')
sb.register_vec('std::vector', 'cv::Mat', excluded=True)
sb.register_ti('cv::MatND')
sb.register_vec('std::vector', 'cv::MatND', excluded=True)

z = sb.register_ti('cv::Rect_', ['int'], 'Rect')
sb.register_vec('std::vector', z, excluded=True)
sb.register_ti('cv::Size_', ['int'], 'Size')
sb.register_ti('cv::TermCriteria')
sb.register_ti('cv::RotatedRect')

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
# highgui.h
#=============================================================================


''')


# Basic GUI functions 
sb.cc.write('''
#-----------------------------------------------------------------------------
# Basic GUI functions 
#-----------------------------------------------------------------------------

    
CV_WINDOW_AUTOSIZE = 1

CV_WND_PROP_FULLSCREEN	 = 0
CV_WND_PROP_AUTOSIZE	 = 1
CV_WINDOW_NORMAL	 	 = 0
CV_WINDOW_FULLSCREEN	 = 1


# Holds references to ctypes function wrappers for callbacks to keep the
# Python side object alive.  Keyed by window name, with a window value being
# a dictionary of callbacks, keyed by "mouse" mouse callback, or "trackbar-name"
# for a trackbar named "name".  
#
# See module bottom for atexit registration to destroy windows at process exit.
_windows_callbacks = {}

# Assigns callback for mouse events
CV_EVENT_MOUSEMOVE = 0
CV_EVENT_LBUTTONDOWN = 1
CV_EVENT_RBUTTONDOWN = 2
CV_EVENT_MBUTTONDOWN = 3
CV_EVENT_LBUTTONUP = 4
CV_EVENT_RBUTTONUP = 5
CV_EVENT_MBUTTONUP = 6
CV_EVENT_LBUTTONDBLCLK = 7
CV_EVENT_RBUTTONDBLCLK = 8
CV_EVENT_MBUTTONDBLCLK = 9

CV_EVENT_FLAG_LBUTTON = 1
CV_EVENT_FLAG_RBUTTON = 2
CV_EVENT_FLAG_MBUTTON = 4
CV_EVENT_FLAG_CTRLKEY = 8
CV_EVENT_FLAG_SHIFTKEY = 16
CV_EVENT_FLAG_ALTKEY = 32

CV_LOAD_IMAGE_UNCHANGED = -1 # 8 bit, color or gray - deprecated, use CV_LOAD_IMAGE_ANYCOLOR
CV_LOAD_IMAGE_GRAYSCALE =  0 # 8 bit, gray
CV_LOAD_IMAGE_COLOR     =  1 # 8 bit unless combined with CV_LOAD_IMAGE_ANYDEPTH, color
CV_LOAD_IMAGE_ANYDEPTH  =  2 # any depth, if specified on its own gray by itself
                             # equivalent to CV_LOAD_IMAGE_UNCHANGED but can be modified
                             # with CV_LOAD_IMAGE_ANYDEPTH
CV_LOAD_IMAGE_ANYCOLOR  =  4

CV_IMWRITE_JPEG_QUALITY = 1
CV_IMWRITE_PNG_COMPRESSION = 16
CV_IMWRITE_PXM_BINARY = 32

CV_CVTIMG_FLIP = 1
CV_CVTIMG_SWAP_RB = 2

CV_CAP_ANY = 0     # autodetect
CV_CAP_MIL = 100     # MIL proprietary drivers
CV_CAP_VFW = 200     # platform native
CV_CAP_V4L = 200
CV_CAP_V4L2 = 200
CV_CAP_FIREWARE = 300     # IEEE 1394 drivers
CV_CAP_FIREWIRE = 300     # IEEE 1394 drivers
CV_CAP_IEEE1394 = 300
CV_CAP_DC1394 = 300
CV_CAP_CMU1394 = 300
CV_CAP_STEREO = 400     # TYZX proprietary drivers
CV_CAP_TYZX = 400
CV_TYZX_LEFT = 400
CV_TYZX_RIGHT = 401
CV_TYZX_COLOR = 402
CV_TYZX_Z = 403
CV_CAP_QT = 500     # Quicktime
CV_CAP_UNICAP = 600   # Unicap drivers
CV_CAP_DSHOW = 700   # DirectShow (via videoInput)
CV_CAP_PVAPI = 800   # PvAPI, Prosilica GigE SDK

CV_CAP_PROP_POS_MSEC      = 0
CV_CAP_PROP_POS_FRAMES    = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH   = 3
CV_CAP_PROP_FRAME_HEIGHT  = 4
CV_CAP_PROP_FPS           = 5
CV_CAP_PROP_FOURCC        = 6
CV_CAP_PROP_FRAME_COUNT   = 7
CV_CAP_PROP_FORMAT        = 8
CV_CAP_PROP_MODE          = 9
CV_CAP_PROP_BRIGHTNESS    =10
CV_CAP_PROP_CONTRAST      =11
CV_CAP_PROP_SATURATION    =12
CV_CAP_PROP_HUE           =13
CV_CAP_PROP_GAIN          =14
CV_CAP_PROP_EXPOSURE      =15
CV_CAP_PROP_CONVERT_RGB   =16
CV_CAP_PROP_WHITE_BALANCE =17
CV_CAP_PROP_RECTIFICATION =18

def CV_FOURCC(c1,c2,c3,c4):
    return (((ord(c1))&255) + (((ord(c2))&255)<<8) + (((ord(c3))&255)<<16) + (((ord(c4))&255)<<24))
    
CV_FOURCC_PROMPT = -1 # Windows only
CV_FOURCC_DEFAULT = CV_FOURCC('I', 'Y', 'U', 'V') # Linux only

''')

# functions
for z in (
    'cvStartWindowThread', 'cvResizeWindow', 'cvMoveWindow', 
    'cvGetWindowName', 
    'cvConvertImage', 
    # 'cvWaitKey', 'cvGrabFrame', 'cvGetCaptureProperty', 'cvSetCaptureProperty', 'cvGetCaptureDomain',
    # 'cvWriteFrame',
    ):
    sb.mb.free_fun(z).include()
    
# CV_FOURCC -- turn it off, we've got ctypes code for it
try:
    sb.mb.free_fun('CV_FOURCC').exclude()
except:
    pass
    
# cvInitSystem
FT.expose_func(sb.mb.free_fun('cvInitSystem'), return_pointee=False, transformer_creators=[
    FT.input_list_of_string('argv', 'argc')])

# cvGetWindowHandle, wait until requested

# setMouseCallback
z = sb.mb.free_fun('cvSetMouseCallback')
FT.expose_func(z, transformer_creators=[FT.mouse_callback_func('on_mouse', 'param')])
FT.add_underscore(z)
sb.cc.write('''
def setMouseCallback(window_name, on_mouse, param=None):
    _windows_callbacks.setdefault(window_name,{})["mouse"] = _ext._cvSetMouseCallback(window_name, on_mouse, param=param)
setMouseCallback.__doc__ = _ext._cvSetMouseCallback.__doc__
''')

# destroyWindow
z = sb.mb.free_fun('cvDestroyWindow')
FT.add_underscore(z)
sb.cc.write('''
def destroyWindow(name):
    _ext._cvDestroyWindow(name)
    if name in _windows_callbacks:
        _windows_callbacks.pop(name)
destroyWindow.__doc__ = _ext._cvDestroyWindow.__doc__        
''')

# destroyAllWindows
z = sb.mb.free_fun('cvDestroyAllWindows')
FT.add_underscore(z)
sb.cc.write('''
def destroyAllWindows():
    _ext._cvDestroyAllWindows()
    _windows_callbacks.clear()
destroyAllWindows.__doc__ = _ext._cvDestroyAllWindows.__doc__        

''')

sb.cc.write('''
# Automatically destroy any remaining tracked windows at process exit,
# otherwise our references to ctypes objects may be destroyed by the normal
# interpreter cleanup before the highgui library cleans up fully, leaving us
# exposed to exceptions.

import atexit
atexit.register(destroyAllWindows)
''')


sb.cc.write('''
#=============================================================================
# highgui.hpp
#=============================================================================


''')


# Basic GUI functions 
sb.cc.write('''
#-----------------------------------------------------------------------------
# C++ Interface
#-----------------------------------------------------------------------------

''')

# functions
for z in (
    'namedWindow', 'imshow', 'imread', 'imwrite', 'imencode', 'imdecode', 'waitKey',
    'setWindowProperty', 'getWindowProperty', 'getTrackbarPos', 'setTrackbarPos',
    ):
    sb.mb.free_fun(z).include()
    
sb.mb.free_fun('imencode')._transformer_creators.append(FT.arg_output('buf'))
    
# createTrackbar
z = sb.mb.free_fun('createTrackbar')
FT.expose_func(z, return_pointee=False, transformer_creators=[
    FT.trackbar_callback2_func('onChange', 'userdata'), FT.from_address('value')])
FT.add_underscore(z)
sb.cc.write('''
def createTrackbar(trackbar_name, window_name, value, count, on_change=None, userdata=None):
    if not isinstance(value, _CT.c_int):
        value = _CT.c_int(value)

    result, z = _ext._createTrackbar(trackbar_name, window_name, _CT.addressof(value), count, on_change, userdata=userdata)
    if result:
        cb_key = 'tracker-' + trackbar_name
        _windows_callbacks.setdefault(window_name,{})[cb_key] = z
    return result
createTrackbar.__doc__ = _ext._createTrackbar.__doc__
''')
sb.add_doc('createTrackbar', "'value' is the initial position of the trackbar. Also, if 'value' is an instance of ctypes.c_int, it holds the current position of the trackbar at any time.")

# VideoCapture
z = sb.mb.class_('VideoCapture')
sb.init_class(z)
z.operator('>>').exclude()
z.add_declaration_code('static cv::VideoCapture &rshift( cv::VideoCapture &inst, cv::Mat &x ){ return inst >> x; }')
z.add_registration_code('def( "__rshift__", &::rshift, bp::return_self<>() )')
sb.finalize_class(z)

# VideoWriter
z = sb.mb.class_('VideoWriter')
sb.init_class(z)
z.operator('<<').exclude()
z.add_declaration_code('static cv::VideoWriter &lshift( cv::VideoWriter &inst, cv::Mat const &x ){ return inst << x; }')
z.add_registration_code('def( "__lshift__", &::lshift, bp::return_self<>() )')
sb.finalize_class(z)
        

sb.done()
