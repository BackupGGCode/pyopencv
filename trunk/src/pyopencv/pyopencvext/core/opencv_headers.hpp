#ifndef SDOPENCV_HEADERS_H
#define SDOPENCV_HEADERS_H

#define __OPENCV_MATRIX_OPERATIONS_H__ // to turn off cxmat.hpp and turn on sdcxmat.hpp -- there's a bug
#include "cxcore.h"
#undef __OPENCV_MATRIX_OPERATIONS_H__
#include "sdcxmat.hpp"

#define __OPENCV_CV_HPP__ // to turn off cv.hpp and turn on sdcv.hpp -- there's a bug
#include "cv.h"
#undef __OPENCV_CV_HPP__
#include "sdcv.hpp"

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
