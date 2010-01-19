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
bool ndarray::iscontiguous() const { check(); return PyArray_ISCONTIGUOUS(ptr()); }

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

// Scalar
VECT_AS_NDARRAY(cv::Scalar, NPY_DOUBLE, 4);

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
    if(obj == Py_None) return object(VectType()); // ndarray = None
    
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

// Scalar
NDARRAY_AS_VECT(cv::Scalar, NPY_DOUBLE, 4);

// Mat
FROM_NDARRAY(cv::Mat)
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
    mat = cv::Mat(cv::Size(shape[1], shape[0]), CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), nchannels), 
        (void *)arr.data(), strides[0]);
    object result(mat);
    objects::make_nurse_and_patient(result.ptr(), obj);
    return result;
}

// MatND
FROM_NDARRAY(cv::MatND)
{
    char s[200];
    
    // checking
    PyObject *obj = arr.ptr();
    if(obj == Py_None) return object(cv::MatND()); // ndarray = None
    
    if(!arr.iscontiguous())
    {
        sprintf(s, "Cannot convert because the ndarray is not contiguous.");
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    
    int sizes[CV_MAX_DIM];    
    int nd = arr.ndim();
    const int *shape = arr.shape();
    for(int i = 0; i < nd; ++i) sizes[i] = shape[nd-1-i];
    
    // wrapping
    CvMatND cvmatnd;
    cvInitMatNDHeader(&cvmatnd, nd, sizes, CV_MAKETYPE(convert_dtype_to_cvdepth(arr.dtype()), 1), 
        (void *)arr.data());
    object result(cv::MatND(&cvmatnd, false));
    objects::make_nurse_and_patient(result.ptr(), obj);
    return result;
}

// ================================================================================================

void mixChannels(const tuple src, tuple dst, const ndarray &fromTo)
{
    char s[200];
    
    const int *shape = fromTo.shape();
    
    if(fromTo.ndim() != 2 || fromTo.dtype() != NPY_LONG || shape[1] != 2 || !fromTo.iscontiguous())
    {
        sprintf(s, "Wrong type! 'fromTo' is not a N-row 2-column int32 C-contiguous ndarray. ");
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
    
    extract<const cv::Mat &> mat(src[0]);
    extract<const cv::MatND &> matnd(src[0]);
    int i, nsrc, ndst;
    if(mat.check())
    {
        std::vector<cv::Mat> src2, dst2;
        nsrc = len(src); src2.resize(nsrc);
        for(i = 0; i < nsrc; ++i) src2[i] = extract<const cv::Mat &>(src[i]);
        ndst = len(dst); dst2.resize(ndst);
        for(i = 0; i < ndst; ++i) dst2[i] = extract<const cv::Mat &>(dst[i]);
        mixChannels(&src2[0], nsrc, &dst2[0], ndst, (const int *)fromTo.data(), shape[0]);
    }
    else if(matnd.check())
    {
        std::vector<cv::MatND> src3, dst3;
        nsrc = len(src); src3.resize(nsrc);
        for(i = 0; i < nsrc; ++i) src3[i] = extract<const cv::MatND &>(src[i]);
        ndst = len(dst); dst3.resize(ndst);
        for(i = 0; i < ndst; ++i) dst3[i] = extract<const cv::MatND &>(dst[i]);
        mixChannels(&src3[0], nsrc, &dst3[0], ndst, (const int *)fromTo.data(), shape[0]);
    }
    else
    {
        sprintf(s, "Cannot determine whether the 1st item of 'src' is Mat or MatND.");
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set();
    }
}

tuple minMaxLoc(const object& a, const object& mask)
{
    double minVal, maxVal;
    int minIdx[CV_MAX_DIM], maxIdx[CV_MAX_DIM];
    int i, n;
    cv::Point minLoc, maxLoc;
    
    tuple result;
    
    extract<const cv::Mat &> mat(a);
    extract<const cv::MatND &> matnd(a);
    extract<const cv::SparseMat &> smat(a);
    if(mat.check())
    {    
        minMaxLoc(mat(), &minVal, &maxVal, &minLoc, &maxLoc, extract<const cv::Mat &>(mask));
        result = make_tuple(object(minVal), object(maxVal), object(minLoc), object(maxLoc));
    }
    else if(matnd.check())
    {
        const cv::MatND &m = matnd();
        minMaxLoc(m, &minVal, &maxVal, minIdx, maxIdx, extract<const cv::MatND &>(mask));
        n = m.dims;
        list l1, l2;
        for(i = 0; i < n; ++i)
        {
            l1.append(object(minIdx[i]));
            l2.append(object(maxIdx[i]));
        }
        result = make_tuple(object(minVal), object(maxVal), tuple(l1), tuple(l2));
    }
    else if(smat.check())
    {
        const cv::SparseMat &m2 = smat();
        minMaxLoc(m2, &minVal, &maxVal, minIdx, maxIdx);
        n = m2.dims();
        list l1, l2;
        for(i = 0; i < n; ++i)
        {
            l1.append(object(minIdx[i]));
            l2.append(object(maxIdx[i]));
        }
        result = make_tuple(object(minVal), object(maxVal), tuple(l1), tuple(l2));
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Cannot determine whether 'a' is Mat, MatND, or SparseMat.");
        throw error_already_set();
    }
    return result;
}


// ================================================================================================

}} // namespace boost::python
