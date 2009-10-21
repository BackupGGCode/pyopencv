#ifndef SDOPENCV_H
#define SDOPENCV_H

// #define CV_NO_BACKWARD_COMPATIBILITY

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


namespace cv {

    typedef DataDepth<bool> DataDepth_bool;
    typedef DataDepth<uchar> DataDepth_uchar;
    typedef DataDepth<schar> DataDepth_schar;
    typedef DataDepth<ushort> DataDepth_ushort;
    typedef DataDepth<short> DataDepth_short;
    typedef DataDepth<int> DataDepth_int;
    typedef DataDepth<float> DataDepth_float;
    typedef DataDepth<double> DataDepth_double;
    
    typedef Rect_<float> Rectf;
    typedef Rect_<double> Rectd;
    
    typedef Size_<double> Size2d;

    CV_INLINE void __dummy_func(
        int __dummy_val = 0
        + sizeof(DataDepth_bool) + sizeof(DataDepth_uchar) + sizeof(DataDepth_schar)
        + sizeof(DataDepth_ushort) + sizeof(DataDepth_short) + sizeof(DataDepth_int)
        + sizeof(DataDepth_float) + sizeof(DataDepth_double) + sizeof(Point2i)
        + sizeof(Point) + sizeof(Complexd) + sizeof(Complexf) + sizeof(Rectf) + sizeof(Rectd)
        + sizeof(Point2d) + sizeof(Vec2d) + sizeof(Size2d) + sizeof(Size) + sizeof(Size2i)
    ) {}
}



#endif
