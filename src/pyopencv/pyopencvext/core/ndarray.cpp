// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

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
        return PyArray_Check(obj) == 1;
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

int ndarray::ndim() const { return PyArray_NDIM(ptr()); }
const Py_intptr_t* ndarray::shape() const { return PyArray_DIMS(ptr()); }
const Py_intptr_t* ndarray::strides() const { return PyArray_STRIDES(ptr()); }
int ndarray::itemsize() const { return PyArray_ITEMSIZE(ptr()); }
int ndarray::dtype() const { return PyArray_TYPE(ptr()); }
const void *ndarray::data() const { return PyArray_DATA(ptr()); }
const void *ndarray::getptr1(int i1) const { return PyArray_GETPTR1(ptr(), i1); }
const void *ndarray::getptr2(int i1, int i2) const { return PyArray_GETPTR2(ptr(), i1, i2); }
const void *ndarray::getptr3(int i1, int i2, int i3) const { return PyArray_GETPTR3(ptr(), i1, i2, i3); }
bool ndarray::iscontiguous() const { return PyArray_ISCONTIGUOUS(ptr()); }

// ================================================================================================

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

// ndarray_to_vector, convert from an ndarray to a std::vector of fixed-size elements
// Note: because Python and C have different ways of allocating/reallocating memory,
// it is UNSAFE to share data between ndarray and std::vector.
// In this implementation, data is allocated and copied instead.

// basic
template<typename Type>
void ndarray_to_vector_basic( const ndarray &in_arr, std::vector<Type> &out_arr )
{
    char s[100];
    int nd = in_arr.ndim();
    if(nd != 1)
    {
        sprintf(s, "Ndarray must be of rank 1, rank %d detected.", nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set(); 
    }
    if(in_arr.dtype() != dtypeof<Type>())
    {
        sprintf(s, "Ndarray's element type is not the same as that of std::vector. ndarray's dtype=%d, vector's dtype=%d.", in_arr.dtype(), dtypeof<Type>());
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set(); 
    }
    
    int len = in_arr.shape()[0];
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = *(Type *)in_arr.getptr1(i);
}

#define NDARRAY_TO_VECTOR_BASIC(Type) \
NDARRAY_TO_VECTOR(Type) { ndarray_to_vector_basic<Type>(in_arr, out_arr); }

NDARRAY_TO_VECTOR_BASIC(char);
NDARRAY_TO_VECTOR_BASIC(unsigned char);
NDARRAY_TO_VECTOR_BASIC(short);
NDARRAY_TO_VECTOR_BASIC(unsigned short);
NDARRAY_TO_VECTOR_BASIC(long);
NDARRAY_TO_VECTOR_BASIC(unsigned long);
NDARRAY_TO_VECTOR_BASIC(int);
NDARRAY_TO_VECTOR_BASIC(unsigned int);
NDARRAY_TO_VECTOR_BASIC(float);
NDARRAY_TO_VECTOR_BASIC(double);

// array
template<typename VectType, int NumpyType, int VectLen>
void ndarray_to_vector_array( const ndarray &in_arr, std::vector<VectType> &out_arr )
{
    char s[100];
    if(!in_arr.ndim()) { out_arr.clear(); return; }

    if(in_arr.dtype() != NumpyType)
    {
        sprintf(s, "Ndarray's element type is not the same as that of std::vector. ndarray's dtype=%d, vector's dtype=%d.", in_arr.dtype(), NumpyType);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set(); 
    }
    
    int len = in_arr.shape()[0];
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = *(VectType *)in_arr.getptr1(i); // this is UNSAFE, but I don't have time to fix yet
}

#define NDARRAY_TO_VECTOR_ARRAY(VectType, NumpyType, VectLen) \
NDARRAY_TO_VECTOR(VectType) { ndarray_to_vector_array<VectType, NumpyType, VectLen>(in_arr, out_arr); }

// Vec-like
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2b, NPY_UBYTE, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3b, NPY_UBYTE, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4b, NPY_UBYTE, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2s, NPY_SHORT, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3s, NPY_SHORT, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4s, NPY_SHORT, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2w, NPY_USHORT, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3w, NPY_USHORT, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4w, NPY_USHORT, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2i, NPY_LONG, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3i, NPY_LONG, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4i, NPY_LONG, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2f, NPY_FLOAT, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3f, NPY_FLOAT, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4f, NPY_FLOAT, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec6f, NPY_FLOAT, 6);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec2d, NPY_DOUBLE, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec3d, NPY_DOUBLE, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec4d, NPY_DOUBLE, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Vec6d, NPY_DOUBLE, 6);

// Point-like
NDARRAY_TO_VECTOR_ARRAY(cv::Point2i, NPY_LONG, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Point2f, NPY_FLOAT, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Point2d, NPY_DOUBLE, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Point3i, NPY_LONG, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Point3f, NPY_FLOAT, 3);
NDARRAY_TO_VECTOR_ARRAY(cv::Point3d, NPY_DOUBLE, 3);

// Rect-like
NDARRAY_TO_VECTOR_ARRAY(cv::Rect, NPY_LONG, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Rectf, NPY_FLOAT, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::Rectd, NPY_DOUBLE, 4);
NDARRAY_TO_VECTOR_ARRAY(cv::RotatedRect, NPY_FLOAT, 5);

// Size-like
NDARRAY_TO_VECTOR_ARRAY(cv::Size2i, NPY_LONG, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Size2f, NPY_FLOAT, 2);
NDARRAY_TO_VECTOR_ARRAY(cv::Size2d, NPY_DOUBLE, 2);

// Scalar
NDARRAY_TO_VECTOR_ARRAY(cv::Scalar, NPY_DOUBLE, 4);

// Range
NDARRAY_TO_VECTOR_ARRAY(cv::Range, NPY_LONG, 2);


// ================================================================================================

// vector_to_ndarray, convert from a std::vector of fixed-size elements to an ndarray

// basic
template<typename Type>
void vector_to_ndarray_basic( const std::vector<Type> &in_arr, ndarray &out_arr )
{
    int len = in_arr.size();
    out_arr = simplenew(1, &len, dtypeof<Type>());
    Type *data = (Type *)out_arr.data();
    for(int i = 0; i < len; ++i) data[i] = in_arr[i];
}

#define VECTOR_TO_NDARRAY_BASIC(Type) \
VECTOR_TO_NDARRAY(Type) { vector_to_ndarray_basic<Type>(in_arr, out_arr); }

VECTOR_TO_NDARRAY_BASIC(char);
VECTOR_TO_NDARRAY_BASIC(unsigned char);
VECTOR_TO_NDARRAY_BASIC(short);
VECTOR_TO_NDARRAY_BASIC(unsigned short);
VECTOR_TO_NDARRAY_BASIC(long);
VECTOR_TO_NDARRAY_BASIC(unsigned long);
VECTOR_TO_NDARRAY_BASIC(int);
VECTOR_TO_NDARRAY_BASIC(unsigned int);
VECTOR_TO_NDARRAY_BASIC(float);
VECTOR_TO_NDARRAY_BASIC(double);

// array
template<typename VectType, int NumpyType, int VectLen>
void vector_to_ndarray_array( const std::vector<VectType> &in_arr, ndarray &out_arr )
{
    int len = in_arr.size();
    int arr[2]; arr[0] = len; arr[1] = VectLen;
    out_arr = simplenew(2, arr, NumpyType);
    VectType *data = (VectType *)out_arr.data();
    for(int i = 0; i < len; ++i) data[i] = in_arr[i];
}

#define VECTOR_TO_NDARRAY_ARRAY(VectType, NumpyType, VectLen) \
VECTOR_TO_NDARRAY(VectType) { vector_to_ndarray_array<VectType, NumpyType, VectLen>(in_arr, out_arr); }

// Vec-like
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2b, NPY_UBYTE, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3b, NPY_UBYTE, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4b, NPY_UBYTE, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2s, NPY_SHORT, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3s, NPY_SHORT, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4s, NPY_SHORT, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2w, NPY_USHORT, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3w, NPY_USHORT, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4w, NPY_USHORT, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2i, NPY_LONG, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3i, NPY_LONG, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4i, NPY_LONG, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2f, NPY_FLOAT, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3f, NPY_FLOAT, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4f, NPY_FLOAT, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec6f, NPY_FLOAT, 6);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec2d, NPY_DOUBLE, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec3d, NPY_DOUBLE, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec4d, NPY_DOUBLE, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Vec6d, NPY_DOUBLE, 6);

// Point-like
VECTOR_TO_NDARRAY_ARRAY(cv::Point2i, NPY_LONG, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Point2f, NPY_FLOAT, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Point2d, NPY_DOUBLE, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Point3i, NPY_LONG, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Point3f, NPY_FLOAT, 3);
VECTOR_TO_NDARRAY_ARRAY(cv::Point3d, NPY_DOUBLE, 3);

// Rect-like
VECTOR_TO_NDARRAY_ARRAY(cv::Rect, NPY_LONG, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Rectf, NPY_FLOAT, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::Rectd, NPY_DOUBLE, 4);
VECTOR_TO_NDARRAY_ARRAY(cv::RotatedRect, NPY_FLOAT, 5);

// Size-like
VECTOR_TO_NDARRAY_ARRAY(cv::Size2i, NPY_LONG, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Size2f, NPY_FLOAT, 2);
VECTOR_TO_NDARRAY_ARRAY(cv::Size2d, NPY_DOUBLE, 2);

// Scalar
VECTOR_TO_NDARRAY_ARRAY(cv::Scalar, NPY_DOUBLE, 4);

// Range
VECTOR_TO_NDARRAY_ARRAY(cv::Range, NPY_LONG, 2);


// ================================================================================================

// as_ndarray -- convert but share data
template<typename VectType, int NumpyType, int VectLen>
ndarray Vect_as_ndarray(const object &obj)
{
    int nd = VectLen;
    ndarray result;
    if(obj.ptr() == Py_None) return result;
    result = new_(1, &nd, NumpyType, 0, (void *)&(extract<const VectType &>(obj)()), 
        NPY_C_CONTIGUOUS | NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.ptr(), obj.ptr());
    return result;
}

#define VECT_AS_NDARRAY(VectType, NumpyType, VectLen) \
AS_NDARRAY(VectType) { return Vect_as_ndarray< VectType, NumpyType, VectLen >(obj); }

// Vec-like
VECT_AS_NDARRAY(cv::Vec2b, NPY_UBYTE, 2);
VECT_AS_NDARRAY(cv::Vec3b, NPY_UBYTE, 3);
VECT_AS_NDARRAY(cv::Vec4b, NPY_UBYTE, 4);
VECT_AS_NDARRAY(cv::Vec2s, NPY_SHORT, 2);
VECT_AS_NDARRAY(cv::Vec3s, NPY_SHORT, 3);
VECT_AS_NDARRAY(cv::Vec4s, NPY_SHORT, 4);
VECT_AS_NDARRAY(cv::Vec2w, NPY_USHORT, 2);
VECT_AS_NDARRAY(cv::Vec3w, NPY_USHORT, 3);
VECT_AS_NDARRAY(cv::Vec4w, NPY_USHORT, 4);
VECT_AS_NDARRAY(cv::Vec2i, NPY_LONG, 2);
VECT_AS_NDARRAY(cv::Vec3i, NPY_LONG, 3);
VECT_AS_NDARRAY(cv::Vec4i, NPY_LONG, 4);
VECT_AS_NDARRAY(cv::Vec2f, NPY_FLOAT, 2);
VECT_AS_NDARRAY(cv::Vec3f, NPY_FLOAT, 3);
VECT_AS_NDARRAY(cv::Vec4f, NPY_FLOAT, 4);
VECT_AS_NDARRAY(cv::Vec6f, NPY_FLOAT, 6);
VECT_AS_NDARRAY(cv::Vec2d, NPY_DOUBLE, 2);
VECT_AS_NDARRAY(cv::Vec3d, NPY_DOUBLE, 3);
VECT_AS_NDARRAY(cv::Vec4d, NPY_DOUBLE, 4);
VECT_AS_NDARRAY(cv::Vec6d, NPY_DOUBLE, 6);

// Point-like
VECT_AS_NDARRAY(cv::Point2i, NPY_LONG, 2);
VECT_AS_NDARRAY(cv::Point2f, NPY_FLOAT, 2);
VECT_AS_NDARRAY(cv::Point2d, NPY_DOUBLE, 2);
VECT_AS_NDARRAY(cv::Point3i, NPY_LONG, 3);
VECT_AS_NDARRAY(cv::Point3f, NPY_FLOAT, 3);
VECT_AS_NDARRAY(cv::Point3d, NPY_DOUBLE, 3);

// Rect-like
VECT_AS_NDARRAY(cv::Rect, NPY_LONG, 4);
VECT_AS_NDARRAY(cv::Rectf, NPY_FLOAT, 4);
VECT_AS_NDARRAY(cv::Rectd, NPY_DOUBLE, 4);
VECT_AS_NDARRAY(cv::RotatedRect, NPY_FLOAT, 5);

// Size-like
VECT_AS_NDARRAY(cv::Size2i, NPY_LONG, 2);
VECT_AS_NDARRAY(cv::Size2f, NPY_FLOAT, 2);
VECT_AS_NDARRAY(cv::Size2d, NPY_DOUBLE, 2);

// Scalar
VECT_AS_NDARRAY(cv::Scalar, NPY_DOUBLE, 4);

// Range
VECT_AS_NDARRAY(cv::Range, NPY_LONG, 2);

// Mat
template<> ndarray as_ndarray<cv::Mat>(const object &obj)
{
    int nd, shape[CV_MAX_DIM], strides[CV_MAX_DIM];
    ndarray result;
    if(obj.ptr() == Py_None) return result;

    cv::Mat mat = extract<const cv::Mat &>(obj)();
    if(!mat.flags) return result; // empty cv::Mat
    
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
    result = new_(nd, shape, convert_cvdepth_to_dtype(mat.depth()), strides, mat.data, NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.ptr(), obj.ptr());
    return result;
}

// MatND
template<> ndarray as_ndarray<cv::MatND>(const object &obj)
{
    int i, nd, shape[CV_MAX_DIM], strides[CV_MAX_DIM];
    ndarray result;
    if(obj.ptr() == Py_None) return result;

    cv::MatND matnd = extract<const cv::MatND &>(obj)();
    if(!matnd.flags) return result; // empty cv::MatND
    
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
    result = new_(nd, shape, convert_cvdepth_to_dtype(matnd.depth()), strides, matnd.data, NPY_WRITEABLE);
    objects::make_nurse_and_patient(result.ptr(), obj.ptr());
    return result;
}

// ================================================================================================

// from_ndarray -- convert but share data
template<typename VectType, int NumpyType, int VectLen>
object ndarray_as_Vect(const ndarray &arr)
{
    char s[200];
    
    // checking
    PyObject *obj = arr.ptr();
    
    int nd = arr.ndim();
    if(nd != 1)
    {
        sprintf(s, "Cannot convert from ndarray to %s because ndim=%d (must be 1).", typeid(VectType).name(), nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    if(arr.dtype() != NumpyType)
    {
        sprintf(s, "Element type must be equivalent to numpy type %d, dtype=%d detected.", NumpyType, arr.dtype());
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    if(!arr.iscontiguous())
    {
        sprintf(s, "The ndarray to be converted must be contiguous .");
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    if(arr.shape()[0] != VectLen)
    {
        sprintf(s, "Number of elements must be %d, shape[0]=%d detected.", VectLen, arr.shape()[0]);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    
    // wrapping
    object result(ptr((VectType *)arr.data()));
    objects::make_nurse_and_patient(result.ptr(), obj);
    return result;
}

#define NDARRAY_AS_VECT(VectType, NumpyType, VectLen) \
FROM_NDARRAY(VectType) { return ndarray_as_Vect< VectType, NumpyType, VectLen >(arr); }

// Vec-like
NDARRAY_AS_VECT(cv::Vec2b, NPY_UBYTE, 2);
NDARRAY_AS_VECT(cv::Vec3b, NPY_UBYTE, 3);
NDARRAY_AS_VECT(cv::Vec4b, NPY_UBYTE, 4);
NDARRAY_AS_VECT(cv::Vec2s, NPY_SHORT, 2);
NDARRAY_AS_VECT(cv::Vec3s, NPY_SHORT, 3);
NDARRAY_AS_VECT(cv::Vec4s, NPY_SHORT, 4);
NDARRAY_AS_VECT(cv::Vec2w, NPY_USHORT, 2);
NDARRAY_AS_VECT(cv::Vec3w, NPY_USHORT, 3);
NDARRAY_AS_VECT(cv::Vec4w, NPY_USHORT, 4);
NDARRAY_AS_VECT(cv::Vec2i, NPY_LONG, 2);
NDARRAY_AS_VECT(cv::Vec3i, NPY_LONG, 3);
NDARRAY_AS_VECT(cv::Vec4i, NPY_LONG, 4);
NDARRAY_AS_VECT(cv::Vec2f, NPY_FLOAT, 2);
NDARRAY_AS_VECT(cv::Vec3f, NPY_FLOAT, 3);
NDARRAY_AS_VECT(cv::Vec4f, NPY_FLOAT, 4);
NDARRAY_AS_VECT(cv::Vec6f, NPY_FLOAT, 6);
NDARRAY_AS_VECT(cv::Vec2d, NPY_DOUBLE, 2);
NDARRAY_AS_VECT(cv::Vec3d, NPY_DOUBLE, 3);
NDARRAY_AS_VECT(cv::Vec4d, NPY_DOUBLE, 4);
NDARRAY_AS_VECT(cv::Vec6d, NPY_DOUBLE, 6);

// Point-like
NDARRAY_AS_VECT(cv::Point2i, NPY_LONG, 2);
NDARRAY_AS_VECT(cv::Point2f, NPY_FLOAT, 2);
NDARRAY_AS_VECT(cv::Point2d, NPY_DOUBLE, 2);
NDARRAY_AS_VECT(cv::Point3i, NPY_LONG, 3);
NDARRAY_AS_VECT(cv::Point3f, NPY_FLOAT, 3);
NDARRAY_AS_VECT(cv::Point3d, NPY_DOUBLE, 3);

// Rect-like
NDARRAY_AS_VECT(cv::Rect, NPY_LONG, 4);
NDARRAY_AS_VECT(cv::Rectf, NPY_FLOAT, 4);
NDARRAY_AS_VECT(cv::Rectd, NPY_DOUBLE, 4);
NDARRAY_AS_VECT(cv::RotatedRect, NPY_FLOAT, 5);

// Size-like
NDARRAY_AS_VECT(cv::Size2i, NPY_LONG, 2);
NDARRAY_AS_VECT(cv::Size2f, NPY_FLOAT, 2);
NDARRAY_AS_VECT(cv::Size2d, NPY_DOUBLE, 2);

// Scalar
NDARRAY_AS_VECT(cv::Scalar, NPY_DOUBLE, 4);

// Range
NDARRAY_AS_VECT(cv::Range, NPY_LONG, 2);

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
        throw error_already_set();
    }
    
    // wrapping
    int cvdepth = CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), nchannels);
    void *data = (void *)arr.data();
    cv::Mat mat;
    if(!nd) mat = cv::Mat(1, 1, cvdepth, data);
    else if(nd == 1) mat = cv::Mat(1, shape[0], cvdepth, data);
    else mat = cv::Mat(shape[nd-1]*strides[nd-1]/strides[1], shape[0], cvdepth, data, strides[1]);
    
    object result(mat);
    objects::make_nurse_and_patient(result.ptr(), arr.ptr());
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
        throw error_already_set();
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
    objects::make_nurse_and_patient(result.ptr(), arr.ptr());
    return result;
}

// ================================================================================================

}} // namespace boost::python
