#ifndef SDOPENCV_CONVERTERS_H
#define SDOPENCV_CONVERTERS_H

#include <cstdio>
#include <vector>
#include <typeinfo>
#include <iostream>

#include <boost/type_traits.hpp>
#include "boost/python.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "boost/python/tuple.hpp"
#include "boost/python/list.hpp"
#include "boost/python/to_python_value.hpp"

#include "opencv_extra.hpp"
#include "sequence.hpp"

// ================================================================================================
// Useful template functions that deal with fixed-size array-like data types

// ------------------------------------------------------------------------------------------------
// is_fixed_size_type : returns true if T is a fixed-size array-like data type
template<typename T>
inline bool is_fixed_size_type() { return false; }

#define DEFINE_FIXED_SIZE(T) \
template<> inline bool is_fixed_size_type< T >() { return true; }

// basic
DEFINE_FIXED_SIZE(char);
DEFINE_FIXED_SIZE(unsigned char);
DEFINE_FIXED_SIZE(short);
DEFINE_FIXED_SIZE(unsigned short);
DEFINE_FIXED_SIZE(long);
DEFINE_FIXED_SIZE(unsigned long);
DEFINE_FIXED_SIZE(int);
DEFINE_FIXED_SIZE(unsigned int);
DEFINE_FIXED_SIZE(float);
DEFINE_FIXED_SIZE(double);

// Vec-like
DEFINE_FIXED_SIZE(cv::Vec2b);
DEFINE_FIXED_SIZE(cv::Vec3b);
DEFINE_FIXED_SIZE(cv::Vec4b);
DEFINE_FIXED_SIZE(cv::Vec2s);
DEFINE_FIXED_SIZE(cv::Vec3s);
DEFINE_FIXED_SIZE(cv::Vec4s);
DEFINE_FIXED_SIZE(cv::Vec2w);
DEFINE_FIXED_SIZE(cv::Vec3w);
DEFINE_FIXED_SIZE(cv::Vec4w);
DEFINE_FIXED_SIZE(cv::Vec2i);
DEFINE_FIXED_SIZE(cv::Vec3i);
DEFINE_FIXED_SIZE(cv::Vec4i);
DEFINE_FIXED_SIZE(cv::Vec2f);
DEFINE_FIXED_SIZE(cv::Vec3f);
DEFINE_FIXED_SIZE(cv::Vec4f);
DEFINE_FIXED_SIZE(cv::Vec6f);
DEFINE_FIXED_SIZE(cv::Vec2d);
DEFINE_FIXED_SIZE(cv::Vec3d);
DEFINE_FIXED_SIZE(cv::Vec4d);
DEFINE_FIXED_SIZE(cv::Vec6d);

// Point-like
DEFINE_FIXED_SIZE(cv::Point2i);
DEFINE_FIXED_SIZE(cv::Point2f);
DEFINE_FIXED_SIZE(cv::Point2d);
DEFINE_FIXED_SIZE(cv::Point3i);
DEFINE_FIXED_SIZE(cv::Point3f);
DEFINE_FIXED_SIZE(cv::Point3d);

// Rect-like
DEFINE_FIXED_SIZE(cv::Rect);
DEFINE_FIXED_SIZE(cv::Rectf);
DEFINE_FIXED_SIZE(cv::Rectd);
DEFINE_FIXED_SIZE(cv::RotatedRect);

// Size-like
DEFINE_FIXED_SIZE(cv::Size2i);
DEFINE_FIXED_SIZE(cv::Size2f);
DEFINE_FIXED_SIZE(cv::Size2d);

// Scalar
DEFINE_FIXED_SIZE(cv::Scalar);

// Range
DEFINE_FIXED_SIZE(cv::Range);


// ------------------------------------------------------------------------------------------------
// elem_type : type of T's elements
template<typename T>
struct elem_type
{
    typedef T type;
};

template<typename T> inline int elem_type_of() { return cvtypeof<typename elem_type<T>::type>(); }

#define ELEM_TYPE(Type, ElemType) template<> struct elem_type<Type> { typedef ElemType type; }

// basic
ELEM_TYPE(bool, bool);
ELEM_TYPE(char, char);
ELEM_TYPE(unsigned char, unsigned char);
ELEM_TYPE(short, short);
ELEM_TYPE(unsigned short, unsigned short);
ELEM_TYPE(int, int);
ELEM_TYPE(unsigned int, unsigned int);
ELEM_TYPE(long, long);
ELEM_TYPE(unsigned long, unsigned long);
ELEM_TYPE(long long, long long);
ELEM_TYPE(unsigned long long, unsigned long long);
ELEM_TYPE(float, float);
ELEM_TYPE(double, double);

// Vec-like
ELEM_TYPE(cv::Vec2b, unsigned char);
ELEM_TYPE(cv::Vec3b, unsigned char);
ELEM_TYPE(cv::Vec4b, unsigned char);
ELEM_TYPE(cv::Vec2s, short);
ELEM_TYPE(cv::Vec3s, short);
ELEM_TYPE(cv::Vec4s, short);
ELEM_TYPE(cv::Vec2w, unsigned short);
ELEM_TYPE(cv::Vec3w, unsigned short);
ELEM_TYPE(cv::Vec4w, unsigned short);
ELEM_TYPE(cv::Vec2i, int);
ELEM_TYPE(cv::Vec3i, int);
ELEM_TYPE(cv::Vec4i, int);
ELEM_TYPE(cv::Vec2f, float);
ELEM_TYPE(cv::Vec3f, float);
ELEM_TYPE(cv::Vec4f, float);
ELEM_TYPE(cv::Vec6f, float);
ELEM_TYPE(cv::Vec2d, double);
ELEM_TYPE(cv::Vec3d, double);
ELEM_TYPE(cv::Vec4d, double);
ELEM_TYPE(cv::Vec6d, double);

// Point-like
ELEM_TYPE(cv::Point2i, int);
ELEM_TYPE(cv::Point2f, float);
ELEM_TYPE(cv::Point2d, double);
ELEM_TYPE(cv::Point3i, int);
ELEM_TYPE(cv::Point3f, float);
ELEM_TYPE(cv::Point3d, double);

// Rect-like
ELEM_TYPE(cv::Rect, int);
ELEM_TYPE(cv::Rectf, float);
ELEM_TYPE(cv::Rectd, double);
ELEM_TYPE(cv::RotatedRect, float);

// Size-like
ELEM_TYPE(cv::Size2i, int);
ELEM_TYPE(cv::Size2f, float);
ELEM_TYPE(cv::Size2d, double);

// Scalar
ELEM_TYPE(cv::Scalar, double);

// Range
ELEM_TYPE(cv::Range, int);


// ------------------------------------------------------------------------------------------------
// n_elems_of : number of elements of T
template<typename T>
inline int n_elems_of()
{
    char s[300];
    sprintf( s, "Instantiation of function n_elems_of() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

// basic
template<> inline int n_elems_of<char>() { return 1; }
template<> inline int n_elems_of<unsigned char>() { return 1; }
template<> inline int n_elems_of<short>() { return 1; }
template<> inline int n_elems_of<unsigned short>() { return 1; }
template<> inline int n_elems_of<long>() { return 1; }
template<> inline int n_elems_of<unsigned long>() { return 1; }
template<> inline int n_elems_of<int>() { return 1; }
template<> inline int n_elems_of<unsigned int>() { return 1; }
template<> inline int n_elems_of<float>() { return 1; }
template<> inline int n_elems_of<double>() { return 1; }

// Vec-like
template<> inline int n_elems_of<cv::Vec2b>() { return 2; }
template<> inline int n_elems_of<cv::Vec3b>() { return 3; }
template<> inline int n_elems_of<cv::Vec4b>() { return 4; }
template<> inline int n_elems_of<cv::Vec2s>() { return 2; }
template<> inline int n_elems_of<cv::Vec3s>() { return 3; }
template<> inline int n_elems_of<cv::Vec4s>() { return 4; }
template<> inline int n_elems_of<cv::Vec2w>() { return 2; }
template<> inline int n_elems_of<cv::Vec3w>() { return 3; }
template<> inline int n_elems_of<cv::Vec4w>() { return 4; }
template<> inline int n_elems_of<cv::Vec2i>() { return 2; }
template<> inline int n_elems_of<cv::Vec3i>() { return 3; }
template<> inline int n_elems_of<cv::Vec4i>() { return 4; }
template<> inline int n_elems_of<cv::Vec2f>() { return 2; }
template<> inline int n_elems_of<cv::Vec3f>() { return 3; }
template<> inline int n_elems_of<cv::Vec4f>() { return 4; }
template<> inline int n_elems_of<cv::Vec6f>() { return 6; }
template<> inline int n_elems_of<cv::Vec2d>() { return 2; }
template<> inline int n_elems_of<cv::Vec3d>() { return 3; }
template<> inline int n_elems_of<cv::Vec4d>() { return 4; }
template<> inline int n_elems_of<cv::Vec6d>() { return 6; }

// Point-like
template<> inline int n_elems_of<cv::Point2i>() { return 2; }
template<> inline int n_elems_of<cv::Point2f>() { return 2; }
template<> inline int n_elems_of<cv::Point2d>() { return 2; }
template<> inline int n_elems_of<cv::Point3i>() { return 3; }
template<> inline int n_elems_of<cv::Point3f>() { return 3; }
template<> inline int n_elems_of<cv::Point3d>() { return 3; }

// Rect-like
template<> inline int n_elems_of<cv::Rect>() { return 4; }
template<> inline int n_elems_of<cv::Rectf>() { return 4; }
template<> inline int n_elems_of<cv::Rectd>() { return 4; }
template<> inline int n_elems_of<cv::RotatedRect>() { return 5; }

// Size-like
template<> inline int n_elems_of<cv::Size2i>() { return 2; }
template<> inline int n_elems_of<cv::Size2f>() { return 2; }
template<> inline int n_elems_of<cv::Size2d>() { return 2; }

// Scalar
template<> inline int n_elems_of<cv::Scalar>() { return 4; }

// Range
template<> inline int n_elems_of<cv::Range>() { return 2; }


// ================================================================================================
// Common converters
template<typename T>
inline void convert_from_vector_to_array(std::vector<T> const &in_arr, T *&out_arr, int &out_len)
{
    out_len = in_arr.size();
    out_arr = &in_arr[0];
}

// ================================================================================================
// New converters related to cv::Mat

// ------------------------------------------------------------------------------------------------
// get the number of elements of type T per row
template<typename T>
int n_elems_per_row( const cv::Mat &mat )
{
    char s[300];
    
    if(mat.empty()) return 0; // empty Mat is valid

    if(mat.depth() != elem_type_of<T>())
    {
        sprintf( s, "Mat's depth (%d) not equal to the element type (%d) of class type '%s'", mat.depth(), elem_type_of<T>(), typeid(T).name() );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    return mat.channels()*mat.cols;
}

// ------------------------------------------------------------------------------------------------
// create a 1-row Mat with n elements of type T, return a pointer to the first element
// if 'in_arr' is not NULL, copy the data to the newly created Mat
template<typename T>
T *create_Mat( cv::Mat &out_arr, int n, T const *in_arr=0 )
{
    if(!n)
    {
        out_arr = cv::Mat();
        return 0;
    }
    
    int nchannels = n_elems_of<T>();
    if(nchannels > 1 && nchannels <= 4) // multi-channel
        out_arr.create(cv::Size(n, 1), CV_MAKETYPE(elem_type_of<T>(), nchannels));
    else
        out_arr.create(cv::Size(nchannels*n, 1), CV_MAKETYPE(elem_type_of<T>(), 1));

    T *ddst = (T *)out_arr.data;
    if(in_arr) for(int i = 0; i < n; ++i) ddst[i] = in_arr[i];
    return ddst;
}

// ------------------------------------------------------------------------------------------------
// convert_from_Mat_to_T
template<typename T>
inline T &convert_from_Mat_to_T( const cv::Mat &in_arr )
{
    char s[300];
    int n = n_elems_per_row<T>(in_arr);
    if(n < n_elems_of<T>())
    {
        sprintf( s, "Mat only has %d elements per row whereas class type '%s' has %d elements.", 
            n, typeid(T).name(), n_elems_of<T>() );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    return *(T *)in_arr.data;
}

template<typename T>
inline void convert_from_Mat_to_T( const cv::Mat &in_arr, T &out_arr )
{
    out_arr = convert_from_Mat_to_T<T>(in_arr);
}

// ------------------------------------------------------------------------------------------------
// convert_from_T_to_Mat
template<typename T>
inline void convert_from_T_to_Mat( const T &in_arr, cv::Mat &out_arr )
{
    *create_Mat<T>(out_arr, 1) = in_arr;
}

template<typename T>
inline cv::Mat convert_from_T_to_Mat( const T &in_arr )
{
    cv::Mat result;
    convert_from_T_to_Mat<T>(in_arr, result);
    return result;
}

// ------------------------------------------------------------------------------------------------
// convert_from_Mat_to_array_of_T
template<typename T>
inline void convert_from_Mat_to_array_of_T( const cv::Mat &in_arr, T *&out_arr, int &out_len )
{    
    out_len = n_elems_per_row<T>(in_arr) / n_elems_of<T>();
    out_arr = out_len? (T *)in_arr.data: 0;
}

// ------------------------------------------------------------------------------------------------
// convert_from_array_of_T_to_Mat
template<typename T>
inline void convert_from_array_of_T_to_Mat( T const *in_arr, int in_len, cv::Mat &out_arr )
{
    if(!out_arr.empty() && out_arr.depth() == elem_type_of<T>()) // same depth
    {
        if(n_elems_per_row<T>(out_arr) == in_len && (T const *)out_arr.data == in_arr)
            return; // same array
    }
    
    create_Mat<T>(out_arr, in_len, in_arr);
}

template<typename T>
inline cv::Mat convert_from_array_of_T_to_Mat( T const *in_arr, int in_len )
{
    cv::Mat result;
    convert_from_array_of_T_to_Mat<T>(in_arr, in_len, result);
    return result;
}

// ------------------------------------------------------------------------------------------------
// convert_from_Mat_to_vector_of_T
template<typename T>
inline void convert_from_Mat_to_vector_of_T( const cv::Mat &in_arr, std::vector<T> &out_arr )
{
    int out_len = n_elems_per_row<T>(in_arr) / n_elems_of<T>();
    if(out_len)
    {
        out_arr.resize(out_len);
        T *out_arr2 = (T *)in_arr.data;
        for(int i = 0; i < out_len; ++i) out_arr[i] = out_arr2[i];
    }
    else
        out_arr.clear();
}

template<typename T>
inline std::vector<T> convert_from_Mat_to_vector_of_T( const cv::Mat &in_arr )
{
    std::vector<T> result;
    convert_from_Mat_to_vector_of_T<T>(in_arr, result);
    return result;
}

// ------------------------------------------------------------------------------------------------
// convert_from_Mat_to_vector_of_vector_of_T
template<typename T>
inline void convert_from_Mat_to_vector_of_vector_of_T( const cv::Mat &in_arr, std::vector<std::vector<T> > &out_arr )
{
    out_arr.resize(in_arr.rows);
    int out_len = n_elems_per_row<T>(in_arr) / n_elems_of<T>();
    for(int r = 0; r < in_arr.rows; ++r)
    {
        out_arr[r].resize(out_len);
        if(out_len)
        {
            T *out_arr3 = &out_arr[r][0];
            T *out_arr2 = (T *)in_arr.ptr(r);
            for(int i = 0; i < out_len; ++i) out_arr3[i] = out_arr2[i];
        }
    }
}

template<typename T>
inline std::vector<std::vector<T> > convert_from_Mat_to_vector_of_vector_of_T( const cv::Mat &in_arr )
{
    std::vector<std::vector<T> > result;
    convert_from_Mat_to_vector_of_vector_of_T<T>(in_arr, result);
    return result;
}

// ------------------------------------------------------------------------------------------------
// convert_from_vector_of_T_to_Mat
template<typename T>
inline void convert_from_vector_of_T_to_Mat( const std::vector<T> &in_arr, cv::Mat &out_arr )
{
    create_Mat<T>(out_arr, in_arr.size(), &in_arr[0]);
}

template<typename T>
inline cv::Mat convert_from_vector_of_T_to_Mat( const std::vector<T> &in_arr )
{
    cv::Mat result;
    convert_from_vector_of_T_to_Mat<T>(in_arr, result);
    return result;
}

// ================================================================================================

// check if a type is a std::vector
template<typename T> inline bool is_std_vector(T *) { return false; }
template<typename T> inline bool is_std_vector(std::vector<T> *) { return true; }
template<typename T> inline bool is_std_vector() { return is_std_vector((typename boost::remove_reference<T>::type *)0); }


// ================================================================================================
// conversion between C++ object and Python object

// ------------------------------------------------------------------------------------------------
// convert_from_T_to_object

// forward declaration
template<typename T>
void convert_from_vector_of_T_to_object(std::vector<T> const &in_arr, bp::object &out_arr);


// convert_from_T_to_object
template<typename T>
inline void convert_from_T_to_object( T const &in_arr, bp::object &out_arr )
{
    bp::extract<T &> out_arr2(out_arr);
    if(!out_arr2.check())
    {
        char s[300];
        sprintf( s, "Unable to convert in function convert_from_T_to_object() because 'out_arr' is not of class type '%s'", typeid(T).name() );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    out_arr2() = in_arr;
}

template<typename T>
inline void convert_from_T_to_object( std::vector<T> const &in_arr, bp::object &out_arr )
{
    convert_from_vector_of_T_to_object(in_arr, out_arr);
}

template<typename T>
inline bp::object convert_from_T_to_object( T const &in_arr )
{
    bp::object obj;
    convert_from_T_to_object(in_arr, obj);
    return obj;
}

template<typename T>
inline bool is_item_fixed_size_type(T *)
{
    char s[300];
    sprintf( s, "Instantiation of function is_item_fixed_size_type<%s>() is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}
template<typename T>
inline bool is_item_fixed_size_type(std::vector<T> *) { return is_fixed_size_type<T>(); }

// convert_from_vector_of_T_to_object
template<typename T>
inline void convert_from_vector_of_T_to_object(std::vector<T> const &in_arr, bp::object &out_arr)
{
    char s[300];
    if(is_fixed_size_type<T>())
    {
        bp::extract<cv::Mat &> out_arr2(out_arr);
        if(!out_arr2.check()) 
        {
            sprintf( s, "Unable to convert in function convert_from_vector_of_T_to_object<%s>() because 'out_arr' is not a Mat object", typeid(T).name() );
            PyErr_SetString(PyExc_TypeError, s);
            throw bp::error_already_set(); 
        }        
        convert_from_vector_of_T_to_Mat(in_arr, out_arr2());
        return;
    }

    int i, n = in_arr.size();
    bp::extract<bp::list> out_arr2(out_arr);
    if(!out_arr2.check())
    {
        sprintf( s, "Unable to convert in function convert_from_vector_of_T_to_object<%s>() because 'out_arr' is not a Python list", typeid(T).name() );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }        
    
    // rectify out_arr
    int m = bp::len(out_arr);
    while(m < n) { out_arr2().append(bp::object()); ++m; }
    while(m > n) { out_arr2().pop(); --m; }
    if(!n) return;
    
    bp::object obj;
    if(!is_std_vector<T>())
    {
        for(i = 0; i < n; ++i)
        {
            obj = bp::extract<T const &>(out_arr[i]).check()? out_arr[i]: bp::object(T());
            convert_from_T_to_object(in_arr[i], obj);
            out_arr[i] = obj;
        }
    }
    else if(is_item_fixed_size_type((T *)0))
    {
        for(i = 0; i < n; ++i)
        {
            obj = bp::extract<cv::Mat const &>(out_arr[i]).check()? out_arr[i]: bp::object(cv::Mat());
            convert_from_T_to_object(in_arr[i], obj);
            out_arr[i] = obj;
        }
    }
    else
    {
        for(i = 0; i < n; ++i)
        {
            obj = bp::extract<bp::list>(out_arr[i]).check()? out_arr[i]: bp::object(bp::list());
            convert_from_T_to_object(in_arr[i], obj);
            out_arr[i] = obj;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// convert_from_object_to_T

// forward declaration
template<typename T>
void convert_from_object_to_vector_of_T( bp::object const &in_arr, std::vector<T> &out_arr );

// convert_from_object_to_T
template<typename T>
inline void convert_from_object_to_T( bp::object const &in_arr, T &out_arr )
{
    T const &out_arr2 = bp::extract<T const &>(in_arr);
    if(&out_arr != &out_arr2) out_arr = out_arr2; // copy if not the same location
}

template<typename T>
inline void convert_from_object_to_T( bp::object const &in_arr, std::vector<T> &out_arr )
{
    convert_from_object_to_vector_of_T(in_arr, out_arr);
}

// convert_from_object_to_vector_of_T
template<typename T>
void convert_from_object_to_vector_of_T( bp::object const &in_arr, std::vector<T> &out_arr )
{
    if(is_fixed_size_type<T>())
    {
        cv::Mat const &mat = bp::extract<cv::Mat const &>(in_arr);
        convert_from_Mat_to_vector_of_T(mat, out_arr);
        return;
    }
    
    int i, n = bp::len(in_arr);
    out_arr.resize(n);
    for(i = 0; i < n; ++i) convert_from_object_to_T(in_arr[i], out_arr[i]);
}

// ================================================================================================

// convert_seq_to_vector
template<typename T>
void convert_seq_to_vector( const sdcpp::sequence &in_seq, std::vector<T> &out_arr )
{
    bp::object const &in_arr = in_seq.get_obj();

    // None
    out_arr.clear();
    if(in_arr.ptr() == Py_None) return;
    
    // ndarray
    bp::extract<sdcpp::ndarray> in_ndarray(in_arr);
    if(in_ndarray.check())
    {
        sdcpp::ndarray_to_vector<T>(in_ndarray(), out_arr);
        return;
    }
    
    // others
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = bp::extract<T>(in_arr[i]);
}

// convert_vector_to_seq
template<typename T>
sdcpp::sequence convert_vector_to_seq( const std::vector<T> &in_arr )
{
    bp::list out_arr;
    int len = in_arr.size();
    if(!len) return sdcpp::sequence(bp::list());
    for(int i = 0; i < len; ++i) out_arr.append(bp::object(in_arr[i]));
    return sdcpp::sequence(out_arr);
}

#define CONVERT_VECTOR_TO_SEQ(Type) template<> sdcpp::sequence convert_vector_to_seq<Type>( const std::vector<Type> &in_arr )

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
CONVERT_VECTOR_TO_SEQ(cv::Rectf);
CONVERT_VECTOR_TO_SEQ(cv::Rectd);
CONVERT_VECTOR_TO_SEQ(cv::RotatedRect);

// Size-like
CONVERT_VECTOR_TO_SEQ(cv::Size2i);
CONVERT_VECTOR_TO_SEQ(cv::Size2f);
CONVERT_VECTOR_TO_SEQ(cv::Size2d);

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
void convert_seq_to_vector_vector( const sdcpp::sequence &in_seq, std::vector < std::vector < T > > &out_arr )
{
    bp::object const &in_arr = in_seq.get_obj();
    out_arr.clear();
    if(in_arr.ptr() == Py_None) return;
    int len = bp::len(in_arr);
    if(!len) return;
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) convert_seq_to_vector(in_arr[i], out_arr[i]);
}

// convert_vector_vector_to_seq
template<typename T>
sdcpp::sequence convert_vector_vector_to_seq( const std::vector < std::vector < T > > &in_arr )
{
    bp::list out_arr;
    int len = in_arr.size();
    if(!len) return sdcpp::sequence(bp::list());
    for(int i = 0; i < len; ++i) out_arr.append(convert_vector_to_seq(in_arr[i]));
    return sdcpp::sequence(out_arr);
}

template<class T>
struct vector_vector_to_python {
    static PyObject* convert(std::vector< std::vector<T> > const &x) {
        return bp::incref(convert_vector_vector_to_seq(x).ptr());
    }
};

// ================================================================================================
// Converters between cv::Mat and bp::list/sdcpp::sequence

// Convert from sdcpp::sequence to cv::Mat
template<typename T>
bp::object convert_from_seq_to_Mat_object(sdcpp::sequence const &in_seq)
{
    bp::object const &in_arr = in_seq.get_obj();
    if(in_arr.ptr() == Py_None) return bp::object(cv::Mat());
    
    bp::extract<sdcpp::ndarray> in_arr2(in_arr);
    if(in_arr2.check()) return sdcpp::from_ndarray<cv::Mat>(in_arr2());
    
    std::vector<T> tmp_arr; convert_seq_to_vector(in_seq, tmp_arr);
    cv::Mat out_arr; convert_from_vector_of_T_to_Mat(tmp_arr, out_arr);
    return bp::object(out_arr);
}

// Convert from cv::Mat to bp::list
template<typename T>
bp::list convert_from_Mat_to_seq(cv::Mat const &in_arr)
{
    bp::list out_arr;
    T *in_arr2; int len_arr; convert_from_Mat_to_array_of_T(in_arr, in_arr2, len_arr);
    for(int i = 0; i < len_arr; ++i) out_arr.append(bp::object(in_arr2[i]));
    return out_arr;
}


// ================================================================================================

template<typename T>
inline std::vector<T> convert_CvSeq_ptr_to_vector(CvSeq *d)
{
    std::vector<T> descriptors;
    if(d)
    {
        descriptors.resize(d->total*d->elem_size/sizeof(T));
        cvCvtSeqToArray(d, &descriptors[0]);
    }
    return descriptors;
}



// ================================================================================================


CvMat * get_CvMat_ptr(cv::Mat const &mat);
IplImage * get_IplImage_ptr(cv::Mat const &mat);

// convert from a sequence of Mat to vector of Mat-equivalent type
// i.e. IplImage, CvMat, IplImage *, CvMat *, cv::Mat, cv::Mat *
template<typename T>
void convert_from_seq_of_Mat_to_vector_of_T(sdcpp::sequence const &in_seq, std::vector<T> &out_arr)
{
    char s[300];
    sprintf( s, "Instantiation of function convert_from_seq_of_Mat_to_vector_of_T() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

#define CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(Type) \
template<> void convert_from_seq_of_Mat_to_vector_of_T(sdcpp::sequence const &in_seq, std::vector<Type> &out_arr)

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(IplImage);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(IplImage *);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(CvMat);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(CvMat *);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Mat);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Mat *);
CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Ptr<cv::Mat>);



// ================================================================================================


CvMatND * get_CvMatND_ptr(cv::MatND const &matnd);


// convert from a sequence of MatND to vector of MatND-equivalent type
// i.e. CvMatND, CvMatND *, cv::MatND, cv::MatND *
template<typename T>
void convert_from_seq_of_MatND_to_vector_of_T(sdcpp::sequence const &in_seq, std::vector<T> &out_arr)
{
    char s[300];
    sprintf( s, "Instantiation of function convert_from_seq_of_MatND_to_vector_of_T() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw bp::error_already_set(); 
}

#define CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(Type) \
template<> void convert_from_seq_of_MatND_to_vector_of_T(sdcpp::sequence const &in_seq, std::vector<Type> &out_arr)

CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(CvMatND);
CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(CvMatND *);
CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(cv::MatND);
CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(cv::MatND *);


// ================================================================================================

template<typename SrcType, typename DstType>
inline DstType normal_cast( SrcType const &inst ) { return inst.operator DstType(); }

// ================================================================================================

template<typename T1, typename T2> inline T1 __add__(T1 const &inst1, T2 const &inst2)
    { return inst1+inst2; }
template<typename T1, typename T2> inline T2 __radd__(T1 const &inst1, T2 const &inst2)
    { return inst1+inst2; }
template<typename T1, typename T2> inline T1 __sub__(T1 const &inst1, T2 const &inst2)
    { return inst1-inst2; }
template<typename T1, typename T2> inline T1 __mul__(T1 const &inst1, T2 const &inst2)
    { return inst1*inst2; }
template<typename T1, typename T2> inline T2 __rmul__(T1 const &inst1, T2 const &inst2)
    { return inst1*inst2; }
template<typename T1, typename T2> inline T1 __div__(T1 const &inst1, T2 const &inst2)
    { return inst1/inst2; }
template<typename T1, typename T2> inline T1 __and__(T1 const &inst1, T2 const &inst2)
    { return inst1&inst2; }
template<typename T1, typename T2> inline T1 __or__(T1 const &inst1, T2 const &inst2)
    { return inst1|inst2; }

template<typename T1> inline T1 __neg__(T1 const &inst)
    { return -inst; }
template<typename T1> inline bool __not__(T1 const &inst)
    { return !inst; }

template<typename T1, typename T2> inline T1 &__iadd__(T1 &inst1, T2 const &inst2)
    { return inst1 += inst2; }
template<typename T1, typename T2> inline T1 &__isub__(T1 &inst1, T2 const &inst2)
    { return inst1 -= inst2; }
template<typename T1, typename T2> inline T1 &__imul__(T1 &inst1, T2 const &inst2)
    { return inst1 *= inst2; }
template<typename T1, typename T2> inline T1 &__idiv__(T1 &inst1, T2 const &inst2)
    { return inst1 /= inst2; }
template<typename T1, typename T2> inline T1 &__iand__(T1 &inst1, T2 const &inst2)
    { return inst1 &= inst2; }
template<typename T1, typename T2> inline T1 &__ior__(T1 &inst1, T2 const &inst2)
    { return inst1 |= inst2; }

template<typename T1, typename T2> inline bool __lt__(T1 const&inst1, T2 const &inst2)
    { return inst1 < inst2; }
template<typename T1, typename T2> inline bool __le__(T1 const&inst1, T2 const &inst2)
    { return inst1 <= inst2; }
template<typename T1, typename T2> inline bool __gt__(T1 const&inst1, T2 const &inst2)
    { return inst1 > inst2; }
template<typename T1, typename T2> inline bool __ge__(T1 const&inst1, T2 const &inst2)
    { return inst1 >= inst2; }
template<typename T1, typename T2> inline bool __ne__(T1 const&inst1, T2 const &inst2)
    { return inst1 != inst2; }
template<typename T1, typename T2> inline bool __eq__(T1 const&inst1, T2 const &inst2)
    { return inst1 == inst2; }

#endif
