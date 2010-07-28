#ifndef SDOPENCV_HEADERS_H
#define SDOPENCV_HEADERS_H

#include "cxcore.h"

namespace cv // missing classes in OpenCV 2.1
{
typedef Size_<double> Size2d;
typedef Rect_<float> Rectf;
typedef Rect_<double> Rectd;
}

template<typename T> inline int _cmp(T const &inst1, T const &inst2)
    { return std::strncmp((char const *)&inst1, (char const *)&inst2, sizeof(T)); }

#define DEFINE_CMP_OPERATORS(T) \
inline bool operator<(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)<0; } \
inline bool operator<=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)<=0; } \
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; } \
inline bool operator!=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)!=0; } \
inline bool operator>=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)>=0; } \
inline bool operator>(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)>0; }

#define DEFINE_EQUAL_OPERATOR(T) \
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }

namespace cv {

template<typename T>
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }

}

template<typename T>
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }


#endif
