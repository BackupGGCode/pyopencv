#ifndef SDOPENCV_HEADERS_H
#define SDOPENCV_HEADERS_H

#include "cxtypes.h"
#include "cxcore.h"
#include "cxoperations.hpp"
#include "cxmat.hpp"
#include "cxcore.hpp"

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

#include "template_instantiations.hpp"


// sdopencv stuff
#include "sdopencv.hpp"


#endif
