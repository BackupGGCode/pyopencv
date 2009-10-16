#ifndef SDOPENCV_H
#define SDOPENCV_H

#define CV_NO_BACKWARD_COMPATIBILITY

#include "cxtypes.h"
#include "cxcore.h"
#include "cvtypes.h"
#define SKIP_INCLUDES
#include "cv.h"
#undef SKIP_INCLUDES
#include "highgui.h"

struct CvGenericHash {};
struct CvFileStorage {};

struct CvHidHaarClassifierCascade {};
struct CvFeatureTree {};
struct CvLSH {};
struct CvLSHOperations {}; // if cv.hpp is not included
struct CvPOSITObject {};

struct CvCapture {};
struct CvVideoWriter {};


#endif
