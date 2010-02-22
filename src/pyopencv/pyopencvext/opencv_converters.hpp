#ifndef SDOPENCV_CONVERTERS_H
#define SDOPENCV_CONVERTERS_H

#include <cstdio>
#include <vector>
#include <typeinfo>

#include "boost/python.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "boost/python/tuple.hpp"
#include "boost/python/list.hpp"
#include "boost/python/to_python_value.hpp"

#include "opencv_extra.hpp"

// ================================================================================================

// convert_Mat
template<typename T>
void convert_Mat( const cv::Mat &in_arr, T &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function convert_Mat() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

template<typename T>
void convert_Mat( const T &in_arr, cv::Mat &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function convert_Mat() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

// convert a Python sequence into a cv::Mat
template<> void convert_Mat<bp::object>( const bp::object &in_arr, cv::Mat &out_arr );


// convert_Mat, T * case
// Only from Mat to T* is implemented. The converse direction is UNSAFE.
template<typename T>
void convert_Mat( const cv::Mat &in_arr, T *&out_arr )
{
    char s[100];
    if(!in_arr.flags) { out_arr = 0; return; }
    
    if(in_arr.rows != 1)
    {
        sprintf(s, "Mat must be a row vector, rows=%d detected.", in_arr.rows);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    if(in_arr.channels() != 1)
    {
        sprintf(s, "Mat must be single-channel, nchannels=%d detected.", in_arr.channels());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    if(in_arr.type() != cvtypeof<T>())
    {
        sprintf(s, "cv::Mat's element type is not the same as that of the output array. cv::Mat's type=%d, vector's type=%d.", in_arr.type(), cvtypeof<T>());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    out_arr = (T *)in_arr.data;
}

extern template void convert_Mat( const cv::Mat &in_arr, char *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, unsigned char *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, short *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, unsigned short *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, long *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, unsigned long *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, int *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, unsigned int *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, float *&out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, double *&out_arr );

// ================================================================================================

// convert_seq_to_vector
template<typename T>
void convert_seq_to_vector( const bp::object &in_arr, std::vector<T> &out_arr )
{
    // None
    out_arr.clear();
    if(in_arr.ptr() == Py_None) return;
    
    // ndarray
    bp::extract<bp::ndarray> in_ndarray(in_arr);
    if(in_ndarray.check())
    {
        bp::ndarray_to_vector<T>(in_ndarray(), out_arr);
        return;
    }
    
    // others
    int len = bp::len(in_arr);
    if(!len) return;
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = bp::extract<T>(in_arr[i]);
}

// convert_vector_to_seq
template<typename T>
bp::sequence convert_vector_to_seq( const std::vector<T> &in_arr )
{
    bp::list out_arr;
    int len = in_arr.size();
    if(!len) return bp::sequence(bp::list());
    for(int i = 0; i < len; ++i) out_arr.append(bp::object(in_arr[i]));
    return bp::sequence(out_arr);
}

#define CONVERT_VECTOR_TO_SEQ(Type) template<> bp::sequence convert_vector_to_seq<Type>( const std::vector<Type> &in_arr )

// basic
CONVERT_VECTOR_TO_SEQ(char);
CONVERT_VECTOR_TO_SEQ(unsigned char);
CONVERT_VECTOR_TO_SEQ(short);
CONVERT_VECTOR_TO_SEQ(unsigned short);
CONVERT_VECTOR_TO_SEQ(long);
CONVERT_VECTOR_TO_SEQ(unsigned long);
CONVERT_VECTOR_TO_SEQ(int);
CONVERT_VECTOR_TO_SEQ(unsigned int);
CONVERT_VECTOR_TO_SEQ(float);
CONVERT_VECTOR_TO_SEQ(double);

// Vec-like
CONVERT_VECTOR_TO_SEQ(cv::Vec2b);
CONVERT_VECTOR_TO_SEQ(cv::Vec3b);
CONVERT_VECTOR_TO_SEQ(cv::Vec4b);
CONVERT_VECTOR_TO_SEQ(cv::Vec2s);
CONVERT_VECTOR_TO_SEQ(cv::Vec3s);
CONVERT_VECTOR_TO_SEQ(cv::Vec4s);
CONVERT_VECTOR_TO_SEQ(cv::Vec2w);
CONVERT_VECTOR_TO_SEQ(cv::Vec3w);
CONVERT_VECTOR_TO_SEQ(cv::Vec4w);
CONVERT_VECTOR_TO_SEQ(cv::Vec2i);
CONVERT_VECTOR_TO_SEQ(cv::Vec3i);
CONVERT_VECTOR_TO_SEQ(cv::Vec4i);
CONVERT_VECTOR_TO_SEQ(cv::Vec2f);
CONVERT_VECTOR_TO_SEQ(cv::Vec3f);
CONVERT_VECTOR_TO_SEQ(cv::Vec4f);
CONVERT_VECTOR_TO_SEQ(cv::Vec6f);
CONVERT_VECTOR_TO_SEQ(cv::Vec2d);
CONVERT_VECTOR_TO_SEQ(cv::Vec3d);
CONVERT_VECTOR_TO_SEQ(cv::Vec4d);
CONVERT_VECTOR_TO_SEQ(cv::Vec6d);

// Point-like
CONVERT_VECTOR_TO_SEQ(cv::Point2i);
CONVERT_VECTOR_TO_SEQ(cv::Point2f);
CONVERT_VECTOR_TO_SEQ(cv::Point2d);
CONVERT_VECTOR_TO_SEQ(cv::Point3i);
CONVERT_VECTOR_TO_SEQ(cv::Point3f);
CONVERT_VECTOR_TO_SEQ(cv::Point3d);

// Rect-like
CONVERT_VECTOR_TO_SEQ(cv::Rect);
CONVERT_VECTOR_TO_SEQ(cv::Rectd);
CONVERT_VECTOR_TO_SEQ(cv::Rectf);
CONVERT_VECTOR_TO_SEQ(cv::RotatedRect);

// Scalar
CONVERT_VECTOR_TO_SEQ(cv::Scalar);

// Range
CONVERT_VECTOR_TO_SEQ(cv::Range);


template<class T>
struct vector_to_python {
    static PyObject* convert(std::vector<T> const &x) {
        return bp::incref(convert_vector_to_seq(x).ptr());
    }
};


// ================================================================================================

// convert_seq_to_vector_vector
template<typename T>
void convert_seq_to_vector_vector( const bp::object &in_arr, std::vector < std::vector < T > > &out_arr )
{
    out_arr.clear();
    if(in_arr.ptr() == Py_None) return;
    int len = bp::len(in_arr);
    if(!len) return;
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) convert_seq_to_vector(in_arr[i], out_arr[i]);
}

// convert_vector_vector_to_seq
template<typename T>
bp::sequence convert_vector_vector_to_seq( const std::vector < std::vector < T > > &in_arr )
{
    bp::list out_arr;
    int len = in_arr.size();
    if(!len) return bp::sequence(bp::list());
    for(int i = 0; i < len; ++i) out_arr.append(convert_vector_to_seq(in_arr[i]));
    return bp::sequence(out_arr);
}

template<class T>
struct vector_vector_to_python {
    static PyObject* convert(std::vector< std::vector<T> > const &x) {
        return bp::incref(convert_vector_vector_to_seq(x).ptr());
    }
};


CvMat * get_CvMat_ptr(cv::Mat &mat);
IplImage * get_IplImage_ptr(cv::Mat &mat);



#endif
