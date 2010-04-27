// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/cast.hpp>
#include <boost/python/ptr.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/detail/raw_pyobject.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/object/life_support.hpp>
#include <arrayobject.h>

#include "ndarray.hpp"
#include "opencv_converters.hpp"

#include <iostream>

namespace bp = boost::python;

// ================================================================================================
// Stuff related to new numpy's ndarray
// ================================================================================================

namespace sdcpp {

// ================================================================================================

void ndarray::check_obj(object const &obj) const
{
    if(!PyArray_Check(obj.ptr())) // not an ndarray
    {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray.");
        throw bp::error_already_set();
    }
}

template<> bool check<ndarray>(object const &obj){ return PyArray_Check(obj.ptr()) == 1; }
template<> PyTypeObject const *get_pytype<ndarray>() { return &PyArray_Type; }

int ndarray::ndim() const { return PyArray_NDIM(obj.ptr()); }
Py_intptr_t ndarray::size() const { return PyArray_SIZE(obj.ptr()); }
const Py_intptr_t* ndarray::shape() const { return PyArray_DIMS(obj.ptr()); }
const Py_intptr_t* ndarray::strides() const { return PyArray_STRIDES(obj.ptr()); }
int ndarray::itemsize() const { return PyArray_ITEMSIZE(obj.ptr()); }
int ndarray::dtype() const { return PyArray_TYPE(obj.ptr()); }
const void *ndarray::data() const { return PyArray_DATA(obj.ptr()); }
const void *ndarray::getptr1(int i1) const { return PyArray_GETPTR1(obj.ptr(), i1); }
const void *ndarray::getptr2(int i1, int i2) const { return PyArray_GETPTR2(obj.ptr(), i1, i2); }
const void *ndarray::getptr3(int i1, int i2, int i3) const { return PyArray_GETPTR3(obj.ptr(), i1, i2, i3); }
bool ndarray::iscontiguous() const { return PyArray_ISCONTIGUOUS(obj.ptr()); }

bool ndarray::last_dim_as_cvchannel() const
{
    int nd = ndim();
    if(!nd) return false;
    
    int nchannels = shape()[nd-1];
    if(nchannels < 1 || nchannels > 4) return false;
    
    int is = itemsize();
    const Py_intptr_t *st = strides();
    if(nd == 1) return itemsize() == st[0];
    
    return is == st[nd-1] && nchannels*is == st[nd-2];
}

int ndarray::cvrank() const { return ndim()-last_dim_as_cvchannel(); }

// ================================================================================================

ndarray simplenew_ndarray(int len, const int *shape, int dtype)
{
    return ndarray(object(handle<>(borrowed(PyArray_SimpleNew(len, 
        (npy_intp *)shape, dtype)))));
}

ndarray new_ndarray(int len, const int *shape, int dtype, const int *strides, void *data, int flags)
{
    return ndarray(object(handle<>(borrowed(PyArray_New(&PyArray_Type, len, 
        (npy_intp *)shape, dtype, (npy_intp *)strides, data, 0, flags, NULL)))));
}

// ================================================================================================

#define DEFVEC(VEC_NAME, ELEM_TYPE, N_ELEM) \
template<> bool check<VEC_NAME>(object const &obj) \
{ \
    if(!check<ndarray>(obj)) return false; \
    ndarray arr(obj); \
    return arr.iscontiguous() && \
        dtype_equiv(arr.dtype(), dtypeof<ELEM_TYPE>()) && \
        (arr.size() % N_ELEM == 0); \
} \
\
template<> PyTypeObject const *get_pytype<VEC_NAME>() { return get_pytype<ndarray>();}

#define DEFVECS(VEC_NAME, ELEM_TYPE) \
DEFVEC(VEC_NAME##_1, ELEM_TYPE, 1) \
DEFVEC(VEC_NAME##_2, ELEM_TYPE, 2) \
DEFVEC(VEC_NAME##_3, ELEM_TYPE, 3) \
DEFVEC(VEC_NAME##_4, ELEM_TYPE, 4) \
DEFVEC(VEC_NAME##_5, ELEM_TYPE, 5) \
DEFVEC(VEC_NAME##_6, ELEM_TYPE, 6)

// DEFVECS(vec_int8, char);
// DEFVECS(vec_uint8, unsigned char);
// DEFVECS(vec_int16, short);
// DEFVECS(vec_uint16, unsigned short);
// DEFVECS(vec_int, int);
// DEFVECS(vec_uint, unsigned int);
// DEFVECS(vec_float32, float);
// DEFVECS(vec_float64, double);

#define REGVECS(VEC_NAME) \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_1>(); \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_2>(); \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_3>(); \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_4>(); \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_5>(); \
sdcpp::register_sdobject<sdcpp::VEC_NAME##_6>();

#define REGVECSS \
REGVECS(vec_int8); \
REGVECS(vec_uint8); \
REGVECS(vec_int16); \
REGVECS(vec_uint16); \
REGVECS(vec_int); \
REGVECS(vec_uint); \
REGVECS(vec_float32); \
REGVECS(vec_float64);

// ================================================================================================

bool dtype_equiv(int dtypenum1, int dtypenum2)
    { return PyArray_EquivTypenums(dtypenum1, dtypenum2); }

// dtypeof

// basic
DTYPEOF(char) { return NPY_BYTE; }
DTYPEOF(unsigned char) { return NPY_UBYTE; }
DTYPEOF(short) { return NPY_SHORT; }
DTYPEOF(unsigned short) { return NPY_USHORT; }
DTYPEOF(long) { return NPY_LONG; }
DTYPEOF(unsigned long) { return NPY_ULONG; }
DTYPEOF(int) { return NPY_INT; }
DTYPEOF(unsigned int) { return NPY_UINT; }
DTYPEOF(long long) { return NPY_LONGLONG; }
DTYPEOF(unsigned long long) { return NPY_ULONGLONG; }
DTYPEOF(float) { return NPY_FLOAT; }
DTYPEOF(double) { return NPY_DOUBLE; }

// ================================================================================================

int convert_dtype_to_cvdepth(int dtype)
{
    switch(dtype)
    {
    case NPY_BYTE: return CV_8S;
    case NPY_UBYTE: return CV_8U;
    case NPY_SHORT: return CV_16S;
    case NPY_USHORT: return CV_16U;
    case NPY_INT: return CV_32S;
    case NPY_LONG:
        if(PyArray_EquivTypenums(NPY_INT, NPY_LONG))
            return CV_32S;
        PyErr_SetString(PyExc_TypeError, "Unconvertable dtype NPY_LONG because it is 64-bit and there is no equivalent CV_64S type.");
        throw bp::error_already_set();
        return -1;
    case NPY_LONGLONG:
        if(PyArray_EquivTypenums(NPY_INT, NPY_LONGLONG))
            return CV_32S;
        PyErr_SetString(PyExc_TypeError, "Unconvertable dtype NPY_LONGLONG because it is 64-bit and there is no equivalent CV_64S type.");
        throw bp::error_already_set();
        return -1;
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
    case CV_32S: return NPY_INT;
    case CV_32F: return NPY_FLOAT;
    case CV_64F: return NPY_DOUBLE;
    }
    PyErr_SetString(PyExc_TypeError, "Unconvertable cvdepth.");
    throw bp::error_already_set();
    return -1;
}

// ================================================================================================

// ndarray_to_vector, convert from an ndarray to a std::vector of fixed-size elements
// Note: because Python and C have different ways of allocating/reallocating memory,
// it is UNSAFE to share data between ndarray and std::vector.
// In this implementation, data is allocated and copied instead.

template<typename T>
void ndarray_to_vector_impl( const ndarray &in_arr, std::vector<T> &out_arr )
{
    char s[100];
    if(!in_arr.ndim()) { out_arr.clear(); return; }
    
    int dtypenum = dtypeof<typename elem_type<T>::type>();
    if(!PyArray_EquivTypenums(in_arr.dtype(), dtypenum))
    {
        sprintf(s, "Ndarray's element type is not the same as that of std::vector. ndarray's dtype=%d, vector's dtype=%d.", in_arr.dtype(), dtypenum);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    int len = in_arr.shape()[0];
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = *(T *)in_arr.getptr1(i);
}

#define NDARRAY_TO_VECTOR_IMPL(T) NDARRAY_TO_VECTOR(T) { ndarray_to_vector_impl< T >(in_arr, out_arr); }

// basic
NDARRAY_TO_VECTOR_IMPL(char);
NDARRAY_TO_VECTOR_IMPL(unsigned char);
NDARRAY_TO_VECTOR_IMPL(short);
NDARRAY_TO_VECTOR_IMPL(unsigned short);
NDARRAY_TO_VECTOR_IMPL(long);
NDARRAY_TO_VECTOR_IMPL(unsigned long);
NDARRAY_TO_VECTOR_IMPL(int);
NDARRAY_TO_VECTOR_IMPL(unsigned int);
NDARRAY_TO_VECTOR_IMPL(float);
NDARRAY_TO_VECTOR_IMPL(double);

// Vec-like
NDARRAY_TO_VECTOR_IMPL(cv::Vec2b);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3b);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4b);
NDARRAY_TO_VECTOR_IMPL(cv::Vec2s);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3s);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4s);
NDARRAY_TO_VECTOR_IMPL(cv::Vec2w);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3w);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4w);
NDARRAY_TO_VECTOR_IMPL(cv::Vec2i);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3i);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4i);
NDARRAY_TO_VECTOR_IMPL(cv::Vec2f);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3f);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4f);
NDARRAY_TO_VECTOR_IMPL(cv::Vec6f);
NDARRAY_TO_VECTOR_IMPL(cv::Vec2d);
NDARRAY_TO_VECTOR_IMPL(cv::Vec3d);
NDARRAY_TO_VECTOR_IMPL(cv::Vec4d);
NDARRAY_TO_VECTOR_IMPL(cv::Vec6d);

// Point-like
NDARRAY_TO_VECTOR_IMPL(cv::Point2i);
NDARRAY_TO_VECTOR_IMPL(cv::Point2f);
NDARRAY_TO_VECTOR_IMPL(cv::Point2d);
NDARRAY_TO_VECTOR_IMPL(cv::Point3i);
NDARRAY_TO_VECTOR_IMPL(cv::Point3f);
NDARRAY_TO_VECTOR_IMPL(cv::Point3d);

// Rect-like
NDARRAY_TO_VECTOR_IMPL(cv::Rect);
NDARRAY_TO_VECTOR_IMPL(cv::Rectf);
NDARRAY_TO_VECTOR_IMPL(cv::Rectd);
NDARRAY_TO_VECTOR_IMPL(cv::RotatedRect);

// Size-like
NDARRAY_TO_VECTOR_IMPL(cv::Size2i);
NDARRAY_TO_VECTOR_IMPL(cv::Size2f);
NDARRAY_TO_VECTOR_IMPL(cv::Size2d);

// Scalar
NDARRAY_TO_VECTOR_IMPL(cv::Scalar);

// Range
NDARRAY_TO_VECTOR_IMPL(cv::Range);


// ================================================================================================

// vector_to_ndarray, convert from a std::vector of fixed-size elements to an ndarray

template<typename T>
void vector_to_ndarray_impl( const std::vector<T> &in_arr, ndarray &out_arr )
{
    int len = in_arr.size();
    int arr[2]; arr[0] = len; arr[1] = n_elems_of<T>();
    out_arr = simplenew_ndarray(arr[1] > 1? 2: 1, arr, dtypeof<typename elem_type<T>::type>());
    T *data = (T *)out_arr.data();
    for(int i = 0; i < len; ++i) data[i] = in_arr[i];
}

#define VECTOR_TO_NDARRAY_IMPL(T) VECTOR_TO_NDARRAY(T) { vector_to_ndarray_impl< T >(in_arr, out_arr); }

// basic
VECTOR_TO_NDARRAY_IMPL(char);
VECTOR_TO_NDARRAY_IMPL(unsigned char);
VECTOR_TO_NDARRAY_IMPL(short);
VECTOR_TO_NDARRAY_IMPL(unsigned short);
VECTOR_TO_NDARRAY_IMPL(long);
VECTOR_TO_NDARRAY_IMPL(unsigned long);
VECTOR_TO_NDARRAY_IMPL(int);
VECTOR_TO_NDARRAY_IMPL(unsigned int);
VECTOR_TO_NDARRAY_IMPL(float);
VECTOR_TO_NDARRAY_IMPL(double);

// Vec-like
VECTOR_TO_NDARRAY_IMPL(cv::Vec2b);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3b);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4b);
VECTOR_TO_NDARRAY_IMPL(cv::Vec2s);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3s);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4s);
VECTOR_TO_NDARRAY_IMPL(cv::Vec2w);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3w);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4w);
VECTOR_TO_NDARRAY_IMPL(cv::Vec2i);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3i);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4i);
VECTOR_TO_NDARRAY_IMPL(cv::Vec2f);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3f);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4f);
VECTOR_TO_NDARRAY_IMPL(cv::Vec6f);
VECTOR_TO_NDARRAY_IMPL(cv::Vec2d);
VECTOR_TO_NDARRAY_IMPL(cv::Vec3d);
VECTOR_TO_NDARRAY_IMPL(cv::Vec4d);
VECTOR_TO_NDARRAY_IMPL(cv::Vec6d);

// Point-like
VECTOR_TO_NDARRAY_IMPL(cv::Point2i);
VECTOR_TO_NDARRAY_IMPL(cv::Point2f);
VECTOR_TO_NDARRAY_IMPL(cv::Point2d);
VECTOR_TO_NDARRAY_IMPL(cv::Point3i);
VECTOR_TO_NDARRAY_IMPL(cv::Point3f);
VECTOR_TO_NDARRAY_IMPL(cv::Point3d);

// Rect-like
VECTOR_TO_NDARRAY_IMPL(cv::Rect);
VECTOR_TO_NDARRAY_IMPL(cv::Rectf);
VECTOR_TO_NDARRAY_IMPL(cv::Rectd);
VECTOR_TO_NDARRAY_IMPL(cv::RotatedRect);

// Size-like
VECTOR_TO_NDARRAY_IMPL(cv::Size2i);
VECTOR_TO_NDARRAY_IMPL(cv::Size2f);
VECTOR_TO_NDARRAY_IMPL(cv::Size2d);

// Scalar
VECTOR_TO_NDARRAY_IMPL(cv::Scalar);

// Range
VECTOR_TO_NDARRAY_IMPL(cv::Range);


// ================================================================================================

// as_ndarray -- convert but share data
template<typename T>
ndarray as_ndarray_impl(const object &obj)
{
    int nd = n_elems_of<T>();
    ndarray result = new_ndarray(1, &nd, dtypeof<typename elem_type<T>::type>(), 
        0, (void *)&(extract<const T &>(obj)()), NPY_C_CONTIGUOUS | NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.get_obj().ptr(), obj.ptr());
    return result;
}

#define AS_NDARRAY_IMPL(T) AS_NDARRAY(T) { return as_ndarray_impl< T >(obj); }

// Vec-like
AS_NDARRAY_IMPL(cv::Vec2b);
AS_NDARRAY_IMPL(cv::Vec3b);
AS_NDARRAY_IMPL(cv::Vec4b);
AS_NDARRAY_IMPL(cv::Vec2s);
AS_NDARRAY_IMPL(cv::Vec3s);
AS_NDARRAY_IMPL(cv::Vec4s);
AS_NDARRAY_IMPL(cv::Vec2w);
AS_NDARRAY_IMPL(cv::Vec3w);
AS_NDARRAY_IMPL(cv::Vec4w);
AS_NDARRAY_IMPL(cv::Vec2i);
AS_NDARRAY_IMPL(cv::Vec3i);
AS_NDARRAY_IMPL(cv::Vec4i);
AS_NDARRAY_IMPL(cv::Vec2f);
AS_NDARRAY_IMPL(cv::Vec3f);
AS_NDARRAY_IMPL(cv::Vec4f);
AS_NDARRAY_IMPL(cv::Vec6f);
AS_NDARRAY_IMPL(cv::Vec2d);
AS_NDARRAY_IMPL(cv::Vec3d);
AS_NDARRAY_IMPL(cv::Vec4d);
AS_NDARRAY_IMPL(cv::Vec6d);

// Point-like
AS_NDARRAY_IMPL(cv::Point2i);
AS_NDARRAY_IMPL(cv::Point2f);
AS_NDARRAY_IMPL(cv::Point2d);
AS_NDARRAY_IMPL(cv::Point3i);
AS_NDARRAY_IMPL(cv::Point3f);
AS_NDARRAY_IMPL(cv::Point3d);

// Rect-like
AS_NDARRAY_IMPL(cv::Rect);
AS_NDARRAY_IMPL(cv::Rectf);
AS_NDARRAY_IMPL(cv::Rectd);
AS_NDARRAY_IMPL(cv::RotatedRect);

// Size-like
AS_NDARRAY_IMPL(cv::Size2i);
AS_NDARRAY_IMPL(cv::Size2f);
AS_NDARRAY_IMPL(cv::Size2d);

// Scalar
AS_NDARRAY_IMPL(cv::Scalar);

// Range
AS_NDARRAY_IMPL(cv::Range);

// Mat
AS_NDARRAY(cv::Mat)
{
    int nd, shape[CV_MAX_DIM], strides[CV_MAX_DIM];
    if(obj.ptr() == Py_None)
    {
        PyErr_SetString(PyExc_TypeError, "'None' cannot be converted into ndarray.");
        throw bp::error_already_set();
    }

    cv::Mat mat = extract<const cv::Mat &>(obj);
    if(!mat.flags)
    {
        PyErr_SetString(PyExc_TypeError, "Empty Mat cannot be converted into ndarray.");
        throw bp::error_already_set();
    }
    
    if(mat.channels() > 1)
    {
        nd = 3;
        shape[0] = mat.rows; shape[1] = mat.cols; shape[2] = mat.channels();
        strides[0] = mat.step; strides[1] = mat.elemSize(); strides[2] = mat.elemSize1();
    }
    else
    {
        nd = 2;
        shape[0] = mat.rows; shape[1] = mat.cols; 
        strides[0] = mat.step; strides[1] = mat.elemSize();
    }
    ndarray result = new_ndarray(nd, shape, convert_cvdepth_to_dtype(mat.depth()), 
        strides, mat.data, NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.get_obj().ptr(), obj.ptr());
    return result;
}

// MatND
AS_NDARRAY(cv::MatND)
{
    int i, nd, shape[CV_MAX_DIM], strides[CV_MAX_DIM];
    if(obj.ptr() == Py_None)
    {
        PyErr_SetString(PyExc_TypeError, "'None' cannot be converted into ndarray.");
        throw bp::error_already_set();
    }

    cv::MatND matnd = extract<const cv::MatND &>(obj)();
    if(!matnd.flags)
    {
        PyErr_SetString(PyExc_TypeError, "Empty MatND cannot be converted into ndarray.");
        throw bp::error_already_set();
    }
    
    nd = matnd.dims;
    for(i = 0; i < nd; ++i)
    {
        shape[i] = matnd.size[nd-1-i];
        strides[i] = matnd.step[nd-1-i];
    }
            
    if(matnd.channels() > 1)
    {
        shape[nd] = matnd.channels();
        strides[nd++] = matnd.elemSize1();
    }
    ndarray result = new_ndarray(nd, shape, convert_cvdepth_to_dtype(matnd.depth()), 
        strides, matnd.data, NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.get_obj().ptr(), obj.ptr());
    return result;
}

// ================================================================================================

// from_ndarray -- convert but share data
template<typename T>
object from_ndarray_impl(const ndarray &arr)
{
    char s[200];
    
    // checking
    PyObject *obj = arr.get_obj().ptr();
    
    int nd = arr.ndim();
    if(nd != 1)
    {
        sprintf(s, "Cannot convert from ndarray to %s because ndim=%d (must be 1).", typeid(T).name(), nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    int dtypenum = dtypeof<typename elem_type<T>::type>();
    if(!PyArray_EquivTypenums(arr.dtype(), dtypenum))
    {
        sprintf(s, "Element type must be equivalent to numpy type %d, dtype=%d detected.", dtypenum, arr.dtype());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    if(!arr.iscontiguous())
    {
        sprintf(s, "The ndarray to be converted must be contiguous .");
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    int len = n_elems_of<T>();
    if(arr.shape()[0] != len)
    {
        sprintf(s, "Number of elements must be %d, shape[0]=%d detected.", len, arr.shape()[0]);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    
    // wrapping
    object result(ptr((T *)arr.data()));
    objects::make_nurse_and_patient(result.ptr(), obj);
    return result;
}

#define FROM_NDARRAY_IMPL(T) FROM_NDARRAY(T) { return from_ndarray_impl< T >(arr); }

// Vec-like
FROM_NDARRAY_IMPL(cv::Vec2b);
FROM_NDARRAY_IMPL(cv::Vec3b);
FROM_NDARRAY_IMPL(cv::Vec4b);
FROM_NDARRAY_IMPL(cv::Vec2s);
FROM_NDARRAY_IMPL(cv::Vec3s);
FROM_NDARRAY_IMPL(cv::Vec4s);
FROM_NDARRAY_IMPL(cv::Vec2w);
FROM_NDARRAY_IMPL(cv::Vec3w);
FROM_NDARRAY_IMPL(cv::Vec4w);
FROM_NDARRAY_IMPL(cv::Vec2i);
FROM_NDARRAY_IMPL(cv::Vec3i);
FROM_NDARRAY_IMPL(cv::Vec4i);
FROM_NDARRAY_IMPL(cv::Vec2f);
FROM_NDARRAY_IMPL(cv::Vec3f);
FROM_NDARRAY_IMPL(cv::Vec4f);
FROM_NDARRAY_IMPL(cv::Vec6f);
FROM_NDARRAY_IMPL(cv::Vec2d);
FROM_NDARRAY_IMPL(cv::Vec3d);
FROM_NDARRAY_IMPL(cv::Vec4d);
FROM_NDARRAY_IMPL(cv::Vec6d);

// Point-like
FROM_NDARRAY_IMPL(cv::Point2i);
FROM_NDARRAY_IMPL(cv::Point2f);
FROM_NDARRAY_IMPL(cv::Point2d);
FROM_NDARRAY_IMPL(cv::Point3i);
FROM_NDARRAY_IMPL(cv::Point3f);
FROM_NDARRAY_IMPL(cv::Point3d);

// Rect-like
FROM_NDARRAY_IMPL(cv::Rect);
FROM_NDARRAY_IMPL(cv::Rectf);
FROM_NDARRAY_IMPL(cv::Rectd);
FROM_NDARRAY_IMPL(cv::RotatedRect);

// Size-like
FROM_NDARRAY_IMPL(cv::Size2i);
FROM_NDARRAY_IMPL(cv::Size2f);
FROM_NDARRAY_IMPL(cv::Size2d);

// Scalar
FROM_NDARRAY_IMPL(cv::Scalar);

// Range
FROM_NDARRAY_IMPL(cv::Range);

// ndarray's shape and strides arrays are big-endian
// OpenCV's MatND's shape and strides arrays are little-endian
void convert_shape_from_ndarray_to_opencv(const ndarray &arr, std::vector<int> &shape, 
    std::vector<int> &strides, int &nchannels, std::vector<bool> &contiguous)
{
    int nd = arr.ndim();
    
    if(!nd)
    {
        shape.clear();
        strides.clear();
        contiguous.clear();
        nchannels = 0; // no element at all
        return;
    }
    
    const Py_intptr_t *arr_shape = arr.shape();
    const Py_intptr_t *arr_strides = arr.strides();
    int arr_itemsize = arr.itemsize();
    
    if(nd==1)
    {
        if(arr_strides[0] == arr_itemsize // is contiguous
            && 1 <= arr_shape[0] && arr_shape[0] <= 4) // with number of items between 1 and 4
        { // this only dimension is a mutil-channel
            shape.clear();
            strides.clear();
            contiguous.clear();
            nchannels = arr_shape[0];
            return;
        }

        // non-contiguous or number of items > 4
        shape.resize(1);
        shape[0] = arr_shape[0];
        strides.resize(1);
        strides[0] = arr_strides[0];
        contiguous.resize(1);
        contiguous[0] = (arr_strides[0] == arr_itemsize);
        nchannels = 1;
        return;
    }
    
    // n >= 2
    if(arr_strides[nd-1] == arr_itemsize // lowest dimension is contiguous
        && 1 <= arr_shape[nd-1] && arr_shape[nd-1] <= 4 // with number of items between 1 and 4
        && arr_strides[nd-2] == arr_itemsize*arr_shape[nd-1]) // second lowest dimension is also contiguous
    { // then lowest dimension is a multi-channel
        nchannels = arr_shape[--nd];
        arr_itemsize *= arr_shape[nd];
    }
    else
        nchannels = 1;
    
    // prepare shape and strides
    int i;
    shape.resize(nd);
    strides.resize(nd);
    for(i = 0; i < nd; ++i)
    {
        shape[i] = arr_shape[nd-1-i];
        strides[i] = arr_strides[nd-1-i];
    }
    
    // prepare contiguous
    contiguous.resize(nd);
    contiguous[0] = (strides[0] == arr_itemsize);
    for(i = 1; i < nd; ++i) contiguous[i] = (strides[i] == strides[i-1]*shape[i-1]);
}


// Mat
FROM_NDARRAY(cv::Mat)
{
    std::vector<int> shape, strides;
    int nd, nchannels;
    std::vector<bool> contiguous;
    convert_shape_from_ndarray_to_opencv(arr, shape, strides, nchannels, contiguous);
    nd = shape.size();
    
    // checking
    for(int i = 0; i < nd; ++i) if(i != 1 && !contiguous[i])
    {
        char s[1000];    
        sprintf(s, "Cannot convert from ndarray to Mat because the dimension %d is not contiguous.", i);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    
    // wrapping
    int cvdepth = CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), nchannels);
    void *data = (void *)arr.data();
    cv::Mat mat;
    if(!nd) mat = cv::Mat(1, 1, cvdepth, data);
    else if(nd == 1) mat = cv::Mat(1, shape[0], cvdepth, data);
    else mat = cv::Mat(shape[nd-1]*strides[nd-1]/strides[1], shape[0], cvdepth, data, strides[1]);
    
    object result(mat);
    objects::make_nurse_and_patient(result.ptr(), arr.get_obj().ptr());
    return result;
}

// MatND
FROM_NDARRAY(cv::MatND)
{
    char s[200];
    
    // checking
    if(!arr.iscontiguous())
    {
        sprintf(s, "Cannot convert because the ndarray is not contiguous.");
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    
    std::vector<int> shape, strides;
    int nd, nchannels;
    std::vector<bool> contiguous;
    convert_shape_from_ndarray_to_opencv(arr, shape, strides, nchannels, contiguous);
    
    if(!shape.size()) { shape.resize(1); shape[0] = 1; }
    nd = shape.size();
    
    // wrapping
    CvMatND cvmatnd;
    cvInitMatNDHeader(&cvmatnd, nd, &shape[0], CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), nchannels), 
        (void *)arr.data());
    object result(cv::MatND(&cvmatnd, false));
    objects::make_nurse_and_patient(result.ptr(), arr.get_obj().ptr());
    return result;
}

// ================================================================================================

static PyArray_ArrFuncs f;

static void
twoint_copyswap(void *dst, void *src, int swap, void *arr)
{
    if (src != NULL) 
	memcpy(dst, src, sizeof(double));
    
    if (swap) {
	register char *a, *b, c;
	a = (char *)dst;
	b = a + 7;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b   = c;	
    }
}

static PyObject *
twoint_getitem(char *ip, PyArrayObject *ap) {
    std::cout << "getitem" << std::endl;
    int *ip2 = (int *)ip;
    cv::Vec2i v(3, 4);
    
    return incref(object(v).ptr());

    npy_int32 a[2];
    
    if ((ap==NULL) || PyArray_ISBEHAVED_RO(ap)) {
	a[0] = *((npy_int32 *)ip);
	a[1] = *((npy_int32 *)ip + 1);
    }
    else {
	ap->descr->f->copyswap(a, ip, !PyArray_ISNOTSWAPPED(ap),
			       ap);
    }
    return Py_None; // Py_BuildValue("(ii)", a[0], a[1]);
}

static int
twoint_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    npy_int32 a[2];
    
    return 0;
    
    if (!PyTuple_Check(op)) {
	PyErr_SetString(PyExc_TypeError, "must be a tuple");
	return -1;
    }
    if (!PyArg_ParseTuple(op, "ii", a, a+1)) return -1;

    if (ap == NULL || PyArray_ISBEHAVED(ap)) {
	memcpy(ov, a, sizeof(double));
    }
    else {
	ap->descr->f->copyswap(ov, a, !PyArray_ISNOTSWAPPED(ap),
			       ap);
    }
    return 0;
}

#define _ALIGN(type) offsetof(struct {char c; type v;},v)



REGISTER_DTYPE(cv::Vec2i)
{
    PyTypeObject *v = (PyTypeObject *)converter::registered_pytype<cv::Vec2i>::get_pytype();
    std::cout << "v=" << v->tp_name << std::endl;
    
    PyArray_Descr *d = PyArray_DescrNewFromType(NPY_INTP);
    // memcpy(&f, d->f, sizeof(f));
    PyArray_InitArrFuncs(&f);
    
    f.copyswap = twoint_copyswap;
    f.getitem = (PyArray_GetItemFunc *)twoint_getitem;
    f.setitem = (PyArray_SetItemFunc *)twoint_setitem;

    std::cout << "d->kind=" << d->kind << std::endl;
    std::cout << "d->type=" << d->type << std::endl;
    
    d->typeobj = v;
    d->f = &f;
    d->hasobject = 0;
    d->type_num = 0;
    d->subarray = 0;
    d->fields = 0;
    // d->hasobject |= NPY_USE_GETITEM;
    // d->hasobject = NPY_USE_GETITEM|NPY_USE_SETITEM;
    // d->hasobject = NPY_ITEM_HASOBJECT | NPY_USE_GETITEM | NPY_USE_SETITEM;
    d->elsize = sizeof(cv::Vec2i);
    d->alignment = _ALIGN(cv::Vec2i);
    int userval = PyArray_RegisterDataType(d);
    std::cout << "userval=" << userval << std::endl;
    
    if(userval == -1)
        throw error_already_set();
    
    d = PyArray_DescrFromType(userval);
    return get_borrowed_object((PyObject *)d);
}

// ================================================================================================

} // namespace sdcpp


// ================================================================================================
// Initialization
// ================================================================================================

void npy_init1()
{
    import_array();
    sdcpp::register_sdobject<sdcpp::ndarray>();
    // REGVECSS;
}

bool npy_init2()
{
    npy_init1();
    return true;
}

bool npy_inited = npy_init2();


