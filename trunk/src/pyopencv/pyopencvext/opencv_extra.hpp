#ifndef SDOPENCV_EXTRA_H
#define SDOPENCV_EXTRA_H

#include <cstdio>
#include <vector>
#include <typeinfo>

#include "boost/python.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "boost/python/tuple.hpp"
#include "boost/python/list.hpp"
#include "boost/python/to_python_value.hpp"

#include "opencv_headers.hpp"
#include "sequence.hpp"
#include "ndarray.hpp"

namespace bp = boost::python;

#define BOOST_PYTHON_MAX_ARITY 30

void CV_CDECL sdTrackbarCallback2(int pos, void* userdata);
void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param);
float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param );

// ================================================================================================

// cvtypeof
template<typename T>
inline int cvtypeof()
{
    char s[300];
    sprintf( s, "Instantiation of function cvtypeof() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

template<> inline int cvtypeof<char>() { return CV_8S; }
template<> inline int cvtypeof<unsigned char>() { return CV_8U; }
template<> inline int cvtypeof<short>() { return CV_16S; }
template<> inline int cvtypeof<unsigned short>() { return CV_16U; }
template<> inline int cvtypeof<long>() { return CV_32S; }
template<> inline int cvtypeof<int>() { return CV_32S; }
template<> inline int cvtypeof<float>() { return CV_32F; }
template<> inline int cvtypeof<double>() { return CV_64F; }

// ================================================================================================

bp::sequence mixChannels(const bp::sequence &src, bp::sequence &dst, const bp::ndarray &fromTo);
bp::tuple minMaxLoc(const bp::object& a, const bp::object& mask=bp::object());


#endif
