#ifndef PYOPENCV_EXT_H
#define PYOPENCV_EXT_H

#include "cxcore.hpp"

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
    
    int __dummy_val = 0
        + sizeof(DataDepth_bool) + sizeof(DataDepth_uchar) + sizeof(DataDepth_schar)
        + sizeof(DataDepth_ushort) + sizeof(DataDepth_short) + sizeof(DataDepth_int)
        + sizeof(DataDepth_float) + sizeof(DataDepth_double) + sizeof(Point2i)
        + sizeof(Point) + sizeof(Complexd) + sizeof(Complexf) + sizeof(Rectf) + sizeof(Rectd)
        + sizeof(Point2d) + sizeof(Vec2d) + sizeof(Size2d) + sizeof(Size) + sizeof(Size2i);
}

#endif
