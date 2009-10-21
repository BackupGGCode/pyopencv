#ifndef SDOPENCV_H
#define SDOPENCV_H

#define SWIG // this was activated when building the official Windows release of OpenCV 2.0

#include "cxtypes.h"
#include "cxcore.h"
#include "cvtypes.h"
#include "cv.h"
#include "cvcompat.h"

#include "highgui.h"
#include "highgui.hpp"

#include "cxflann.h"

#include "ml.h"

#include "cxoperations.hpp"
#include "cxmat.hpp"
#include "cxcore.hpp"
#include "cv.hpp"

#include "cvaux.hpp"
#include "cvvidsurv.hpp"
#include "cvaux.h"

struct CvGenericHash {};
struct CvFileStorage {};

struct _CvContourScanner {};
struct CvHidHaarClassifierCascade {};
struct CvFeatureTree {};
struct CvLSH {};
// struct CvLSHOperations {}; // if cv.hpp is not included
struct CvPOSITObject {};

struct CvCapture {};
struct CvVideoWriter {};


void CV_CDECL sdTrackbarCallback2(int pos, void* userdata);
void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param);


namespace flann {
    class Index {};
}

#include "template_instantiations.h"


#endif
