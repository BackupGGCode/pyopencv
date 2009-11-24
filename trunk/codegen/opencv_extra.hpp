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



template<typename T>
void convert_ndarray_to( const bp::object &in_arr, T &out_arr )
{
    const char message[] = "Instantiation function convert_ndarray_to() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> void convert_ndarray_to< cv::Mat >( const bp::object &in_arr, cv::Mat &out_arr );

template<> void convert_ndarray_to< std::vector<int> >( const bp::object &in_arr, std::vector<int> &out_arr );



template<typename T>
void convert_ndarray_from( const T &in_arr, bp::object &out_arr )
{
    const char message[] = "Instantiation function convert_ndarray_from() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> void convert_ndarray_from< std::vector<uchar> >( const std::vector<uchar> &in_arr, bp::object &out_arr );


#endif
