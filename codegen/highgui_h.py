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
# highgui.h
#=============================================================================


    ''')


    # Basic GUI functions 
    cc.write('''
#-----------------------------------------------------------------------------
# Basic GUI functions 
#-----------------------------------------------------------------------------

    
CV_WINDOW_AUTOSIZE = 1

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
        'cvStartWindowThread', 'cvShowImage', 'cvResizeWindow', 'cvMoveWindow', 
        'cvGetWindowName', 'cvGetTrackbarPos', 'cvSetTrackbarPos',
        'cvConvertImage', 'cvWaitKey',
        'cvGrabFrame', 'cvGetCaptureProperty', 'cvSetCaptureProperty', 'cvGetCaptureDomain',
        'cvWriteFrame',
        ):
        mb.free_fun(z).include()
        
    for z in ('CvCapture', 'CvVideoWriter'):
        mb.class_(z).include()
    mb.insert_del_interface('CvCapture', '_PE._cvReleaseCapture')
    mb.insert_del_interface('CvVideoWriter', '_PE._cvReleaseVideoWriter')
        
    # TODO: fix these functions:
    # cvInitSystem, cvGetWindowHandle, cvCreateTrackbar, cvCreateTrackbar2, cvSetMouseCallback

    # cvSetMouseCallback
    z = mb.free_fun('cvSetMouseCallback')
    FT.expose_func(z, transformer_creators=[FT.mouse_callback_func('on_mouse', 'param')])
    FT.add_underscore(z)
    cc.write('''
def cvSetMouseCallback(window_name, on_mouse, param=None):
    _windows_callbacks.setdefault(window_name,{})["mouse"] = _PE._cvSetMouseCallback(window_name, on_mouse, param=param)
cvSetMouseCallback.__doc__ = _PE._cvSetMouseCallback.__doc__
    ''')

    # cvNamedWindow
    z = mb.free_fun('cvNamedWindow')
    FT.add_underscore(z)
    cc.write('''
def cvNamedWindow(name, flags=1):
    """int cvNamedWindow(string name, int flags)

    Creates window
    """
    z = _PE._cvNamedWindow(name, flags=flags)
    if z > 0 and not name in _windows_callbacks:
        _windows_callbacks[name] = {}
    return z
    
    ''')
    
    for z in (
        'cvLoadImage', 'cvLoadImageM', 'cvDecodeImage', 'cvDecodeImageM',
        ):
        FT.expose_func(mb.free_fun(z), ownershiplevel=3)

    # cvSaveImage # TODO: fix

    # cvEncodeImage # TODO: fix

    # cvDestroyWindow
    z = mb.free_fun('cvDestroyWindow')
    FT.add_underscore(z)
    cc.write('''
def cvDestroyWindow(name):
    """void cvDestroyWindow(string name)

    Destroys a window
    """
    _PE._cvDestroyWindow(name)
    if name in _windows_callbacks:
        _windows_callbacks.pop(name)
        
    ''')

    # cvDestroyAllWindows
    z = mb.free_fun('cvDestroyAllWindows')
    FT.add_underscore(z)
    cc.write('''
def cvDestroyAllWindows():
    """void cvDestroyAllWindows(void)

    Destroys all the HighGUI windows
    """
    _PE._cvDestroyAllWindows()
    _windows_callbacks.clear()

    ''')


    cc.write('''
# Automatically destroy any remaining tracked windows at process exit,
# otherwise our references to ctypes objects may be destroyed by the normal
# interpreter cleanup before the highgui library cleans up fully, leaving us
# exposed to exceptions.

import atexit
atexit.register(cvDestroyAllWindows)
    ''')

    for z in ('cvRetrieveFrame', 'cvQueryFrame'):
        FT.expose_func(mb.free_fun(z), ward_indices=(1,))

    for z in ('cvCreateFileCapture', 'cvCreateCameraCapture', 'cvCreateVideoWriter'):
        FT.expose_func(mb.free_fun(z))


