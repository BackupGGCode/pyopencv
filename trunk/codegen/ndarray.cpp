// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/handle.hpp>
#include <boost/python/cast.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/detail/raw_pyobject.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/object/life_support.hpp>
#include <arrayobject.h>

#include "ndarray.hpp"

namespace bp = boost::python;

namespace boost { namespace python {

// ================================================================================================
// Stuff related to numpy's ndarray
// ================================================================================================


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

// dtypeof
template<> int dtypeof<char>() { return NPY_BYTE; }
template<> int dtypeof<unsigned char>() { return NPY_UBYTE; }
template<> int dtypeof<short>() { return NPY_SHORT; }
template<> int dtypeof<unsigned short>() { return NPY_USHORT; }
template<> int dtypeof<long>() { return NPY_LONG; }
template<> int dtypeof<unsigned long>() { return NPY_ULONG; }
template<> int dtypeof<int>() { return sizeof(int) == 4? NPY_LONG : NPY_LONGLONG; }
template<> int dtypeof<unsigned int>() { return sizeof(int) == 4? NPY_ULONG : NPY_ULONGLONG; }
template<> int dtypeof<float>() { return NPY_FLOAT; }
template<> int dtypeof<double>() { return NPY_DOUBLE; }

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
    throw error_already_set();
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
    throw error_already_set();
    return -1;
}

// ================================================================================================

namespace aux
{
    bool array_object_manager_traits::check(PyObject* obj)
    {
        return obj == Py_None || PyArray_Check(obj) == 1; // None or ndarray
    }

    python::detail::new_non_null_reference
    array_object_manager_traits::adopt(PyObject* obj)
    {
        return detail::new_non_null_reference(
        pytype_check(&PyArray_Type, obj));
    }

    PyTypeObject const* array_object_manager_traits::get_pytype()
    {
        return &PyArray_Type;
    }
}

void ndarray::check() const
{
    if(PyArray_Check(ptr()) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "The variable is not an ndarray.");
        throw error_already_set(); 
    }
}

int ndarray::ndim() const { check(); return PyArray_NDIM(ptr()); }
const int* ndarray::shape() const { check(); return PyArray_DIMS(ptr()); }
const int* ndarray::strides() const { check(); return PyArray_STRIDES(ptr()); }
int ndarray::itemsize() const { check(); return PyArray_ITEMSIZE(ptr()); }
int ndarray::dtype() const { check(); return PyArray_TYPE(ptr()); }
const void *ndarray::data() const { check(); return PyArray_DATA(ptr()); }
const void *ndarray::getptr1(int i1) const { check(); return PyArray_GETPTR1(ptr(), i1); }
const void *ndarray::getptr2(int i1, int i2) const { check(); return PyArray_GETPTR2(ptr(), i1, i2); }
const void *ndarray::getptr3(int i1, int i2, int i3) const { check(); return PyArray_GETPTR3(ptr(), i1, i2, i3); }

// ================================================================================================

bool ndarray::last_dim_as_cvchannel() const
{
    check();
    
    int nd = ndim();
    if(!nd) return false;
    
    int nchannels = shape()[nd-1];
    if(nchannels < 1 || nchannels > 4) return false;
    
    int is = itemsize();
    const int *st = strides();
    if(nd == 1) return itemsize() == st[0];
    
    return is == st[nd-1] && nchannels*is == st[nd-2];
}

int ndarray::cvrank() const { check(); return ndim()-last_dim_as_cvchannel(); }

// ================================================================================================

ndarray simplenew(int len, const int *shape, int dtype)
{
    return extract<ndarray>(object(handle<>(PyArray_SimpleNew(len, (npy_intp *)shape, dtype))));
}

ndarray new_(int len, const int *shape, int dtype, const int *strides, void *data, int flags)
{
    return extract<ndarray>(object(handle<>(PyArray_New(&PyArray_Type, len, (npy_intp *)shape, 
        dtype, (npy_intp *)strides, data, 0, flags, NULL))));
}

// ================================================================================================

template<> void convert_ndarray< cv::Scalar >( const ndarray &in_arr, cv::Scalar &out_arr )
{
    if(in_arr.dtype() != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Input element type is not double.");
        throw error_already_set(); 
    }
    if(in_arr.ndim() != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Input is not 1D.");
        throw error_already_set(); 
    }
    int len = in_arr.shape()[0];
    if(len > 4) len = 4;
    while(len-- > 0) out_arr[len] = *(double *)in_arr.getptr1(len);
}

// ================================================================================================

template<> void convert_ndarray< cv::Scalar >( const cv::Scalar &in_arr, ndarray &out_arr )
{
    int len = 4;
    out_arr = simplenew(1, &len, NPY_DOUBLE);
    double *data = (double *)out_arr.data();
    while(len-- > 0) data[len] = in_arr[len];
}

// ================================================================================================

template<> void convert_ndarray< cv::Mat >( const ndarray &in_arr, cv::Mat &out_arr )
{
    // PyObject *arr = in_arr.ptr();
    // char s[100];
    // if(PyArray_Check(arr) != 1)
    // {
        // PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        // throw bp::error_already_set(); 
    // }
    // bool lindex_is_channel = last_index_is_channel(in_arr);
    // int nd = PyArray_NDIM(arr);
    // if(nd != 2+lindex_is_channel)
    // {
        // sprintf( s, "Rank must be 2+last_index_is_channel. Detected rank=%d and last_index_is_channel=%d.", nd, lindex_is_channel);
        // PyErr_SetString(PyExc_TypeError, s);
        // throw bp::error_already_set(); 
    // }
    
    // int nchannels;
    // int *shape = PyArray_DIMS(arr);
    // int itemsize = PyArray_ITEMSIZE(arr);
    // int *strides = PyArray_STRIDES(arr);
    
    // if(nd == 2)
    // {
        // if(strides[1] != itemsize) // non-contiguous
        // {
            // sprintf(s, "The last (2nd) dimension must be contiguous (last stride=%d and itemsize=%d).", strides[1], itemsize);
            // PyErr_SetString(PyExc_TypeError, s);
            // throw bp::error_already_set(); 
        // }
        // nchannels = 1;
    // }
    // else
    // {
        // if(strides[2] != itemsize) // non-contiguous
        // {
            // sprintf(s, "The last (3rd) dimension must be contiguous (last stride=%d and itemsize=%d).", strides[2], itemsize);
            // PyErr_SetString(PyExc_TypeError, s);
            // throw bp::error_already_set(); 
        // }
        // nchannels = shape[2];
        // if(strides[1] != itemsize*nchannels) // non-contiguous
        // {
            // sprintf(s, "The 2nd dimension must be contiguous (2nd stride=%d, itemsize=%d, nchannels=%d).", strides[1], itemsize, nchannels);
            // throw bp::error_already_set(); 
        // }
    // }
    // out_arr = cv::Mat(cv::Size(shape[1], shape[0]), 
        // CV_MAKETYPE(convert_dtype_to_cvdepth(PyArray_TYPE(arr)), nchannels), PyArray_DATA(arr), strides[0]);
}

// ================================================================================================

// bool is_Mat_same_shape_with_ndarray( const cv::Mat &in_arr, bp::object &out_arr )
// {
    // PyObject *arr = out_arr.ptr();
    // if(PyArray_Check(arr) != 1) return false;
    // int nd = PyArray_NDIM(arr);
    // if(nd < 2 || nd > 3) return false;
    // int nchannels;
    // int *shape = PyArray_DIMS(arr);
    // int itemsize = PyArray_ITEMSIZE(arr);
    // int *strides = PyArray_STRIDES(arr);
    // if(nd == 2)
    // {
        // if(strides[1] != itemsize) return false;
        // nchannels = 1;
    // }
    // else
    // {
        // if(strides[2] != itemsize) return false;
        // nchannels = shape[2];
        // if(nchannels < 1 || nchannels > 4 || strides[1] != itemsize*nchannels) return false;
    // }
    // if(in_arr.cols != shape[1] || in_arr.rows != shape[0] || in_arr.step != strides[0] ||
        // in_arr.channels() != nchannels || in_arr.depth() != convert_dtype_to_cvdepth(PyArray_TYPE(arr)))
        // return false;
    // return true;
// }

// TODO: later I will create a function to wrap around a cv::Mat, for the case of VideoCapture in highgui
template<> void convert_ndarray< cv::Mat >( const cv::Mat &in_arr, ndarray &out_arr )
{
    // PyObject *arr;
    // int rows = in_arr.rows, cols = in_arr.cols, nchannels = in_arr.channels();
    // int i, rowlen = cols*in_arr.elemSize();
    // if(is_Mat_same_shape_with_ndarray(in_arr, out_arr)) arr = out_arr.ptr();
    // else
    // {
        // int shape[3];
        // shape[0] = rows; shape[1] = cols;    
        // if(nchannels == 1)
            // arr = PyArray_SimpleNew(2, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        // else
        // {
            // shape[2] = nchannels;
            // arr = PyArray_SimpleNew(3, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        // }
        // out_arr = bp::object(bp::handle<>(arr));
    // }
    
    // if(PyArray_DATA(arr) != (void *)in_arr.data)
    // {
        // for(i = 0; i < rows; ++i)
            // std::memmove(PyArray_GETPTR1(arr, i), (const void *)in_arr.ptr(i), rowlen);
    // }
    // else // do nothing
        // std::cout << "Same data location. No copy was needed." << std::endl;
}

// ================================================================================================

template<> void convert_ndarray< cv::MatND >( const ndarray &in_arr, cv::MatND &out_arr )
{
    // PyObject *arr = in_arr.ptr();
    // char s[100];
    // if(PyArray_Check(arr) != 1)
    // {
        // PyErr_SetString(PyExc_TypeError, "Input argument is not an ndarray.");
        // throw bp::error_already_set(); 
    // }
    // if(PyArray_ISCONTIGUOUS(arr) != true)
    // {
        // sprintf(s, "Cannot convert the ndarray into a cv::MatND because it is not C-style contiguous.");
        // PyErr_SetString(PyExc_TypeError, s);
        // throw bp::error_already_set(); 
    // }
    
    // bool lindex_is_channel = last_index_is_channel(in_arr);
    // int *shape = PyArray_DIMS(arr);
    // int nd = PyArray_NDIM(arr);
    // int nchannels = lindex_is_channel? shape[--nd]: 1;
    
    // int rshape[CV_MAX_DIM];    
    // for(int i = 0; i < nd; ++i) rshape[i] = shape[nd-1-i];
    
    // CvMatND cvmatnd;
    // cvInitMatNDHeader( &cvmatnd, nd, rshape, CV_MAKETYPE(convert_dtype_to_cvdepth(PyArray_TYPE(arr)), nchannels), PyArray_DATA(arr) );
    
    // out_arr = cv::MatND(&cvmatnd, false);
}

// ================================================================================================

// bool is_MatND_same_shape_with_ndarray( const cv::MatND &in_arr, bp::object &out_arr )
// {
    // PyObject *arr = out_arr.ptr();
    // if(PyArray_Check(arr) != 1 || PyArray_ISCONTIGUOUS(arr) != true || PyArray_ITEMSIZE(arr) != in_arr.elemSize1()) 
        // return false;
        
    // bool lindex_is_channel = last_index_is_channel(out_arr);
    // int *shape = PyArray_DIMS(arr);
    // int nd = PyArray_NDIM(arr);
    // int nchannels = lindex_is_channel? shape[--nd]: 1;
    // if(nchannels != in_arr.channels()) return false;
    
    // for(int i = 0; i < nd; ++i) if(shape[i] != in_arr.size[nd-1-i]) return false;
    
    // return true;
// }

template<> void convert_ndarray< cv::MatND >( const cv::MatND &in_arr, ndarray &out_arr )
{
    // PyObject *arr;
    // if(is_MatND_same_shape_with_ndarray(in_arr, out_arr)) arr = out_arr.ptr();
    // else
    // {
        // int nd = in_arr.dims, shape[CV_MAX_DIM];
        // for(int i = 0; i < nd; ++i) shape[i] = in_arr.size[nd-1-i];
        // int nchannels = in_arr.channels();
        // if(nchannels > 1) shape[nd++] = nchannels;
        // arr = PyArray_SimpleNew(nd, shape, convert_cvdepth_to_dtype(in_arr.depth()));
        
        // out_arr = bp::object(bp::handle<>(arr));
    // }
    
    // if(PyArray_DATA(arr) != (void *)in_arr.data)
    // {
        // int count = in_arr.step[in_arr.dims-1]*in_arr.size[in_arr.dims-1];
        // std::memmove(PyArray_DATA(arr), (const void *)in_arr.data, count);
    // }
    // else do nothing
}

// ================================================================================================

template void convert_ndarray( const ndarray &in_arr, std::vector<char> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned char> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<short> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned short> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<long> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned long> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<int> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned int> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<float> &out_arr );
template void convert_ndarray( const ndarray &in_arr, std::vector<double> &out_arr );

// ================================================================================================

template void convert_ndarray( const std::vector<char> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<unsigned char> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<short> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<unsigned short> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<long> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<unsigned long> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<int> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<unsigned int> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<float> &in_arr, ndarray &out_arr );
template void convert_ndarray( const std::vector<double> &in_arr, ndarray &out_arr );

// ================================================================================================

ndarray as_ndarray(const object &obj)
{
    int nd, shape[CV_MAX_DIM], strides[CV_MAX_DIM];
    ndarray result;
    if(obj.ptr() == Py_None) return result;

    extract<const cv::Scalar &> scalar(obj);
    extract<const cv::Mat &> mat(obj);
    if(scalar.check())
    {
        nd = 4;
        result = new_(1, &nd, NPY_DOUBLE, 0, (void *)&scalar().val[0], NPY_C_CONTIGUOUS | NPY_WRITEABLE);
    } else if(mat.check())
    {
        cv::Mat mat2 = mat();
        if(!mat2.flags) return result; // empty cv::Mat
        
        if(mat2.channels() > 1)
        {
            nd = 3;
            shape[0] = mat2.rows; shape[1] = mat2.cols; shape[2] = mat2.channels();
            strides[0] = mat2.step; strides[1] = mat2.elemSize(); strides[2] = mat2.elemSize1();
        }
        else
        {
            nd = 2;
            shape[0] = mat2.rows; shape[1] = mat2.cols; 
            strides[0] = mat2.step; strides[1] = mat2.elemSize();
        }
        result = new_(nd, shape, convert_cvdepth_to_dtype(mat2.depth()), strides, mat2.data, NPY_WRITEABLE);
    }
    objects::make_nurse_and_patient(result.ptr(), obj.ptr());
    return result;
}

// ================================================================================================

object as_Scalar(const ndarray &arr)
{
    return object();
}

// ================================================================================================

object as_Mat(const ndarray &arr)
{
    cv::Mat mat;
    char s[1000];
    
    // checking
    PyObject *obj = arr.ptr();
    if(obj == Py_None) return object(mat); // ndarray = None
    
    int nd = arr.ndim();
    if(nd < 2 || nd > 3)
    {
        sprintf(s, "Cannot convert from ndarray to Mat because ndim=%d, expecting 2 (single-channel) or 3 (multiple-channel) only.", nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    int nchannels;
    const int *shape = arr.shape();
    int itemsize = arr.itemsize();
    const int *strides = arr.strides();
    if(nd == 2)
    {
        if(strides[1] != itemsize)
        {
            sprintf(s, "Cannot convert from ndarray to Mat because the last (2nd) dimension is not contiguous: strides[1]=%d, itemsize=%d.", strides[1], itemsize);
            PyErr_SetString(PyExc_TypeError, s);
            throw error_already_set();
        }
        nchannels = 1;
    }
    else
    {
        if(strides[2] != itemsize)
        {
            sprintf(s, "Cannot convert from ndarray to Mat because the last (3rd) dimension is not contiguous: strides[2]=%d, itemsize=%d.", strides[2], itemsize);
            PyErr_SetString(PyExc_TypeError, s);
            throw error_already_set();
        }
        nchannels = shape[2];
        if(nchannels < 1 || nchannels > 4)
        {
            sprintf(s, "Cannot convert from ndarray to Mat because the number of channels is not between 1 and 4 (nchannels=%d).", nchannels);
            PyErr_SetString(PyExc_TypeError, s);
            throw error_already_set();
        }
        if(strides[1] != itemsize*nchannels)
        {
            sprintf(s, "Cannot convert from ndarray to Mat because the second last (2nd) dimension is not contiguous: strides[2]=%d, itemsize=%d, nchannels=%d.", strides[2], itemsize, nchannels);
            PyErr_SetString(PyExc_TypeError, s);
            throw error_already_set();
        }
    }
    
    // wrapping
    mat = cv::Mat(cv::Size(shape[0], shape[1]), CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), nchannels), 
        (void *)arr.data(), strides[0]);
    object result(mat);
    objects::make_nurse_and_patient(result.ptr(), obj);
    return result;
}

// ================================================================================================

object as_MatND(const ndarray &arr)
{
    return object();
}

// ================================================================================================

}} // namespace boost::python
