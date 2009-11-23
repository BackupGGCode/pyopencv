#include "opencv_extra.hpp"

#include <iostream>
#include <cstdio>
#include <string>

#include <boost/python/extract.hpp>

namespace bp = boost::python;

void CV_CDECL sdTrackbarCallback2(int pos, void* userdata)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)userdata)));
    (items[0])(pos, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}


void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)param)));
    (items[0])(event, x, y, flags, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}

float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param )
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)user_param)));
    // pass 'a' and 'b' by address instead of by pointer
    return bp::extract < float >((items[0])((int)a, (int)b, bp::object(items[1]))); // need a copy of items[1] to make it safe with threading
}

int get_cvdepth_from_dtype(bp::object dtype)
{
    const std::string s(bp::extract<const char *>(dtype.attr("name")));
    if(s == "int8") return CV_8S;
    if(s == "uint8") return CV_8U;
    if(s == "int16") return CV_16S;
    if(s == "uint16") return CV_16U;
    if(s == "int32") return CV_32S;
    if(s == "float32") return CV_32F;
    if(s == "float64") return CV_64F;
    PyErr_SetString(PyExc_TypeError, "Unconvertable dtype.");
    throw bp::error_already_set();
    return -1;
}

template<> void convert_ndarray_to< cv::Mat >( const bp::numeric::array &in_arr, cv::Mat &out_matr )
{
    bp::object shape = in_arr.attr("shape");
    int nd = bp::len(shape);
    if(nd < 2)
    {
        PyErr_SetString(PyExc_TypeError, "Rank must not be less than 2.");
        throw bp::error_already_set(); 
    }
    if(nd > 3)
    {
        PyErr_SetString(PyExc_TypeError, "Rank must not be greater than 3.");
        throw bp::error_already_set(); 
    }
    
    int nchannels;    
    int itemsize = bp::extract<int>(in_arr.attr("itemsize"));
    bp::object strides = in_arr.attr("strides");
    
    if(nd == 2)
    {
        nchannels = 1;
        if(bp::extract<int>(strides[1]) != itemsize) // non-contiguous
        {
            PyErr_SetString(PyExc_TypeError, "The last (2nd) dimension must be contiguous.");
            throw bp::error_already_set(); 
        }
    }
    else
    {
        if(bp::extract<int>(strides[2]) != itemsize) // non-contiguous
        {
            PyErr_SetString(PyExc_TypeError, "The last (3rd) dimension must be contiguous.");
            throw bp::error_already_set(); 
        }
        nchannels = bp::extract<int>(shape[2]);
        if(nchannels < 1) // non-contiguous
        {
            PyErr_SetString(PyExc_TypeError, "The number of channels must not be less than 1.");
            throw bp::error_already_set(); 
        }
        if(nchannels > 4) // non-contiguous
        {
            PyErr_SetString(PyExc_TypeError, "The number of channels must not be greater than 4.");
            throw bp::error_already_set(); 
        }
        if(bp::extract<int>(strides[1]) != itemsize*nchannels) // non-contiguous
        {
            PyErr_SetString(PyExc_TypeError, "The 2nd dimension must be contiguous.");
            throw bp::error_already_set(); 
        }
    }
    out_matr = cv::Mat(cv::Size(bp::extract<int>(shape[0]), bp::extract<int>(shape[1])), 
        CV_MAKETYPE(get_cvdepth_from_dtype(in_arr.attr("dtype")), nchannels), 
        (void *)(int)(bp::extract<int>(in_arr.attr("ctypes").attr("data"))), 
        bp::extract<int>(strides[0]));
}
