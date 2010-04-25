#ifndef SDOPENCV_HEADERS_H
#define SDOPENCV_HEADERS_H

#define __OPENCV_MATRIX_OPERATIONS_H__ // to turn off cxmat.hpp and turn on sdcxmat.hpp -- there's a bug
#include "cxcore.h"
#undef __OPENCV_MATRIX_OPERATIONS_H__
#include "sdcxmat.hpp"

#include "cvtypes.h"
#include "cv.h"
#include "cvcompat.h"
#include "cv.hpp"

#include "highgui.h"
#include "highgui.hpp"

#include "cxflann.h"

#include "ml.h"


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

namespace flann {
    class Index {};
}


// sdopencv stuff
#include "sdopencv.hpp"

#include "template_instantiations.hpp"

#endif
