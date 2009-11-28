#include "opencv_extra.hpp"

#include <iostream>
#include <cstdio>
#include <string>

#include <boost/python/extract.hpp>

// ================================================================================================

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

int get_cvdepth_from_dtype(int dtype)
{
    switch(dtype)
    {
    case NPY_BYTE: return CV_8S;
    case NPY_UBYTE: return CV_8U;
    case NPY_SHORT: return CV_16S;
    case NPY_USHORT: return CV_16U;
    case NPY_LONG: return CV_32S;
    case NPY_FLOAT: return CV_32F;
    case NPY_DOUBLE: return CV_64F;
    }
    PyErr_SetString(PyExc_TypeError, "Unconvertable dtype.");
    throw bp::error_already_set();
    return -1;
}

// ================================================================================================


void npy_init1()
{
    import_array();
}

bool npy_init2()
{
    npy_init1();
    return true;
}

bool npy_inited = npy_init2();

// ================================================================================================



template<> void convert_ndarray_to< cv::Mat >( const bp::object &in_arr, cv::Mat &out_arr )
{
    PyObject *arr = in_arr.ptr();
    char s[100];
    if(PyArray_Check(arr) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        throw bp::error_already_set(); 
    }
    int nd = PyArray_NDIM(arr);
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
    int *shape = PyArray_DIMS(arr);
    int itemsize = PyArray_ITEMSIZE(arr);
    int *strides = PyArray_STRIDES(arr);
    
    if(nd == 2)
    {
        if(strides[1] != itemsize) // non-contiguous
        {
            sprintf(s, "The last (2nd) dimension must be contiguous (last stride=%d and itemsize=%d).", strides[1], itemsize);
            PyErr_SetString(PyExc_TypeError, s);
            throw bp::error_already_set(); 
        }
        nchannels = 1;
    }
    else
    {
        if(strides[2] != itemsize) // non-contiguous
        {
            sprintf(s, "The last (3rd) dimension must be contiguous (last stride=%d and itemsize=%d).", strides[2], itemsize);
            PyErr_SetString(PyExc_TypeError, s);
            throw bp::error_already_set(); 
        }
        nchannels = shape[2];
        if(nchannels < 1) // non-contiguous
        {
            sprintf(s, "The number of channels must not be less than 1 (nchannels=%d).", nchannels);
            PyErr_SetString(PyExc_TypeError, s);
            throw bp::error_already_set(); 
        }
        if(nchannels > 4) // non-contiguous
        {
            sprintf(s, "The number of channels must not be greater than 4 (nchannels=%d).", nchannels);
            PyErr_SetString(PyExc_TypeError, s);
            throw bp::error_already_set(); 
        }
        if(strides[1] != itemsize*nchannels) // non-contiguous
        {
            sprintf(s, "The 2nd dimension must be contiguous (2nd stride=%d, itemsize=%d, nchannels=%d).", strides[1], itemsize, nchannels);
            throw bp::error_already_set(); 
        }
    }
    out_arr = cv::Mat(cv::Size(shape[1], shape[0]), 
        CV_MAKETYPE(get_cvdepth_from_dtype(PyArray_TYPE(arr)), nchannels), PyArray_DATA(arr), strides[0]);
}

template void convert_ndarray_to( const bp::object &in_arr, std::vector<char> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned char> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<short> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned short> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<long> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned long> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<int> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<unsigned int> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<float> &out_arr );
template void convert_ndarray_to( const bp::object &in_arr, std::vector<double> &out_arr );

// ================================================================================================


template void convert_ndarray_from( const std::vector<char> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<unsigned char> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<short> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<unsigned short> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<long> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<unsigned long> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<int> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<unsigned int> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<float> &in_arr, bp::object &out_arr );
template void convert_ndarray_from( const std::vector<double> &in_arr, bp::object &out_arr );


