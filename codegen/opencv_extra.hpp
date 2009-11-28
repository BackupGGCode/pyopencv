#ifndef SDOPENCV_EXTRA_H
#define SDOPENCV_EXTRA_H

#include <vector>

#include "opencv_headers.hpp"

#include "boost/python.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "boost/python/tuple.hpp"

#include <arrayobject.h>

namespace bp = boost::python;

CV_INLINE CvPyramid sdCreatePyramid( const CvArr* img, int extra_layers, double rate,
                                const CvSize* layer_sizes CV_DEFAULT(0),
                                CvArr* bufarr CV_DEFAULT(0),
                                int calc CV_DEFAULT(1),
                                int filter CV_DEFAULT(CV_GAUSSIAN_5x5) )
{
    CvPyramid pyr;
    pyr.pyramid = cvCreatePyramid(img, extra_layers, rate, layer_sizes, bufarr, calc, filter);
    pyr.extra_layers = extra_layers;
    return pyr;
}


void CV_CDECL sdTrackbarCallback2(int pos, void* userdata);
void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param);
float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param );



// ================================================================================================
// Stuff related to numpy's ndarray
// ================================================================================================

// dtypeof
template<typename T>
inline int dtypeof()
{
    const char message[] = "Instantiation of function dtypeof() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> inline int dtypeof<char>() { return NPY_BYTE; }
template<> inline int dtypeof<unsigned char>() { return NPY_UBYTE; }
template<> inline int dtypeof<short>() { return NPY_SHORT; }
template<> inline int dtypeof<unsigned short>() { return NPY_USHORT; }
template<> inline int dtypeof<long>() { return NPY_LONG; }
template<> inline int dtypeof<unsigned long>() { return NPY_ULONG; }
template<> inline int dtypeof<int>() { return sizeof(int) == 4? NPY_LONG : NPY_LONGLONG; }
template<> inline int dtypeof<unsigned int>() { return sizeof(int) == 4? NPY_ULONG : NPY_ULONGLONG; }
template<> inline int dtypeof<float>() { return NPY_FLOAT; }
template<> inline int dtypeof<double>() { return NPY_DOUBLE; }


// convert_ndarray_to
template<typename T>
void convert_ndarray_to( const bp::object &in_arr, T &out_arr )
{
    const char message[] = "Instantiation of function convert_ndarray_to() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> void convert_ndarray_to< cv::Mat >( const bp::object &in_arr, cv::Mat &out_arr );

// convert_ndarray_to, std::vector case
template<typename T>
void convert_ndarray_to( const bp::object &in_arr, std::vector<T> &out_arr )
{
    PyObject *arr = in_arr.ptr();
    char s[100];
    if(PyArray_Check(arr) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        throw bp::error_already_set(); 
    }
    int nd = PyArray_NDIM(arr);
    if(nd != 1)
    {
        sprintf(s, "Rank must be 1, rank=%d detected.", nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    if(PyArray_TYPE(arr) != dtypeof<T>())
    {
        sprintf(s, "Ndarray's element type is not the same as that of std::vector. ndarray's dtype=%d, vector's dtype=%d.", PyArray_TYPE(arr), dtypeof<T>());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    int len = PyArray_DIM(arr, 0);
    
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = *(T *)PyArray_GETPTR1(arr, i);
}

extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<char> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned char> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<short> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned short> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<long> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned long> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<int> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned int> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<float> &out_arr );
extern template void convert_ndarray_to( const bp::object &in_arr, std::vector<double> &out_arr );


// convert_ndarray_from
template<typename T>
void convert_ndarray_from( const T &in_arr, bp::object &out_arr )
{
    const char message[] = "Instantiation function convert_ndarray_from() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> void convert_ndarray_from< std::vector<uchar> >( const std::vector<uchar> &in_arr, bp::object &out_arr );


#endif
