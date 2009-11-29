#include <boost/python/detail/prefix.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/object.hpp>

#include <algorithm>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>

#include "opencv_extra.hpp"


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

// ================================================================================================

int convert_dtype_to_cvdepth(int dtype)
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

int convert_cvdepth_to_dtype(int depth)
{
    switch(depth)
    {
    case CV_8S: return NPY_BYTE;
    case CV_8U: return NPY_UBYTE;
    case CV_16S: return NPY_SHORT;
    case CV_16U: return NPY_USHORT;
    case CV_32S: return NPY_LONG;
    case CV_32F: return NPY_FLOAT;
    case CV_64F: return NPY_DOUBLE;
    }
    PyErr_SetString(PyExc_TypeError, "Unconvertable cvdepth.");
    throw bp::error_already_set();
    return -1;
}

// ================================================================================================

// last_index_is_channel
bool last_index_is_channel(const bp::object &in_arr)
{
    PyObject *arr = in_arr.ptr();
    if(PyArray_Check(arr) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        throw bp::error_already_set(); 
    }
    
    int nd = PyArray_NDIM(arr);
    if(!nd) return false;
    
    int nchannels = PyArray_DIM(arr, nd-1);
    if(nchannels < 1 || nchannels > 4) return false;
    
    int itemsize = PyArray_ITEMSIZE(arr);
    int *strides = PyArray_STRIDES(arr);
    if(nd == 1) return itemsize == strides[0];
    
    return itemsize == strides[nd-1] && nchannels*itemsize == strides[nd-2];
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

template<> void convert_ndarray< cv::Mat >( const bp::object &in_arr, cv::Mat &out_arr )
{
    PyObject *arr = in_arr.ptr();
    char s[100];
    if(PyArray_Check(arr) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        throw bp::error_already_set(); 
    }
    bool lindex_is_channel = last_index_is_channel(in_arr);
    int nd = PyArray_NDIM(arr);
    if(nd != 2+lindex_is_channel)
    {
        sprintf( s, "Rank must be 2+last_index_is_channel. Detected rank=%d and last_index_is_channel=%d.", nd, lindex_is_channel);
        PyErr_SetString(PyExc_TypeError, s);
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
        if(strides[1] != itemsize*nchannels) // non-contiguous
        {
            sprintf(s, "The 2nd dimension must be contiguous (2nd stride=%d, itemsize=%d, nchannels=%d).", strides[1], itemsize, nchannels);
            throw bp::error_already_set(); 
        }
    }
    out_arr = cv::Mat(cv::Size(shape[1], shape[0]), 
        CV_MAKETYPE(convert_dtype_to_cvdepth(PyArray_TYPE(arr)), nchannels), PyArray_DATA(arr), strides[0]);
}

// ================================================================================================

bool is_Mat_same_shape_with_ndarray( const cv::Mat &in_arr, bp::object &out_arr )
{
    PyObject *arr = out_arr.ptr();
    if(PyArray_Check(arr) != 1) return false;
    int nd = PyArray_NDIM(arr);
    if(nd < 2 || nd > 3) return false;
    int nchannels;
    int *shape = PyArray_DIMS(arr);
    int itemsize = PyArray_ITEMSIZE(arr);
    int *strides = PyArray_STRIDES(arr);
    if(nd == 2)
    {
        if(strides[1] != itemsize) return false;
        nchannels = 1;
    }
    else
    {
        if(strides[2] != itemsize) return false;
        nchannels = shape[2];
        if(nchannels < 1 || nchannels > 4 || strides[1] != itemsize*nchannels) return false;
    }
    if(in_arr.cols != shape[1] || in_arr.rows != shape[0] || in_arr.step != strides[0] ||
        in_arr.channels() != nchannels || in_arr.depth() != convert_dtype_to_cvdepth(PyArray_TYPE(arr)))
        return false;
    return true;
}

// TODO: later I will create a function to wrap around a cv::Mat, for the case of VideoCapture in highgui
template<> void convert_ndarray< cv::Mat >( const cv::Mat &in_arr, bp::object &out_arr )
{
    PyObject *arr;
    int rows = in_arr.rows, cols = in_arr.cols, nchannels = in_arr.channels();
    int i, rowlen = cols*in_arr.elemSize();
    if(is_Mat_same_shape_with_ndarray(in_arr, out_arr)) arr = out_arr.ptr();
    else
    {
        int shape[3];
        shape[0] = rows; shape[1] = cols;    
        if(nchannels == 1)
            arr = PyArray_SimpleNew(2, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        else
        {
            shape[2] = nchannels;
            arr = PyArray_SimpleNew(3, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        }
        out_arr = bp::object(bp::handle<>(arr));
    }
    
    if(PyArray_DATA(arr) != (void *)in_arr.data)
    {
        for(i = 0; i < rows; ++i)
            std::memmove(PyArray_GETPTR1(arr, i), (const void *)in_arr.ptr(i), rowlen);
    }
    else
        std::cout << "Same data location. No copy was needed." << std::endl;
}

// ================================================================================================

template<> void convert_ndarray< cv::MatND >( const bp::object &in_arr, cv::MatND &out_arr )
{
    PyObject *arr = in_arr.ptr();
    char s[100];
    if(PyArray_Check(arr) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        throw bp::error_already_set(); 
    }
    if(PyArray_ISCONTIGUOUS(arr) != true)
    {
        sprintf(s, "Cannot convert the ndarray into a cv::MatND because it is not C-style contiguous.");
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    bool lindex_is_channel = last_index_is_channel(in_arr);
    int *shape = PyArray_DIMS(arr);
    int nd = PyArray_NDIM(arr);
    int nchannels = lindex_is_channel? shape[--nd]: 1;
    
    int rshape[CV_MAX_DIM];    
    for(int i = 0; i < nd; ++i) rshape[i] = shape[nd-1-i];
    
    CvMatND cvmatnd;
    cvInitMatNDHeader( &cvmatnd, nd, rshape, CV_MAKETYPE(convert_dtype_to_cvdepth(PyArray_TYPE(arr)), nchannels), PyArray_DATA(arr) );
    
    out_arr = cv::MatND(&cvmatnd, false);
}

// ================================================================================================

bool is_MatND_same_shape_with_ndarray( const cv::MatND &in_arr, bp::object &out_arr )
{
    PyObject *arr = out_arr.ptr();
    if(PyArray_Check(arr) != 1 || PyArray_ISCONTIGUOUS(arr) != true || PyArray_ITEMSIZE(arr) != in_arr.elemSize1()) 
        return false;
        
    bool lindex_is_channel = last_index_is_channel(out_arr);
    int *shape = PyArray_DIMS(arr);
    int nd = PyArray_NDIM(arr);
    int nchannels = lindex_is_channel? shape[--nd]: 1;
    if(nchannels != in_arr.channels()) return false;
    
    for(int i = 0; i < nd; ++i) if(shape[i] != in_arr.size[nd-1-i]) return false;
    
    return true;
}

template<> void convert_ndarray< cv::MatND >( const cv::MatND &in_arr, bp::object &out_arr )
{
    PyObject *arr;
    if(is_MatND_same_shape_with_ndarray(in_arr, out_arr)) arr = out_arr.ptr();
    else
    {
        int nd = in_arr.dims, shape[CV_MAX_DIM];
        for(int i = 0; i < nd; ++i) shape[i] = in_arr.size[nd-1-i];
        int nchannels = in_arr.channels();
        if(nchannels > 1) shape[nd++] = nchannels;
        arr = PyArray_SimpleNew(nd, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        
        out_arr = bp::object(bp::handle<>(arr));
    }
    
    if(PyArray_DATA(arr) != (void *)in_arr.data)
    {
        int count = in_arr.step[in_arr.dims-1]*in_arr.size[in_arr.dims-1];
        std::memmove(PyArray_DATA(arr), (const void *)in_arr.data, count);
    }
    // else do nothing
}

// ================================================================================================

template void convert_ndarray( const bp::object &in_arr, std::vector<char> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<unsigned char> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<short> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<unsigned short> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<long> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<unsigned long> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<int> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<unsigned int> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<float> &out_arr );
template void convert_ndarray( const bp::object &in_arr, std::vector<double> &out_arr );

// ================================================================================================

template void convert_ndarray( const std::vector<char> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<unsigned char> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<short> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<unsigned short> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<long> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<unsigned long> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<int> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<unsigned int> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<float> &in_arr, bp::object &out_arr );
template void convert_ndarray( const std::vector<double> &in_arr, bp::object &out_arr );

// ================================================================================================

// get_ndarray_type
PyTypeObject const* get_ndarray_type()
{
    return &PyArray_Type;
}
