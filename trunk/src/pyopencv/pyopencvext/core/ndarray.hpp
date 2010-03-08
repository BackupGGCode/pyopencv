// Copyright Minh-Tri Pham 2010.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef SD_NDARRAY_HPP
# define SD_NDARRAY_HPP

# include <boost/python/detail/prefix.hpp>

# include <boost/python/tuple.hpp>
# include <boost/python/str.hpp>
# include <boost/preprocessor/iteration/local.hpp>
# include <boost/preprocessor/cat.hpp>
# include <boost/preprocessor/repetition/enum.hpp>
# include <boost/preprocessor/repetition/enum_params.hpp>
# include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include "opencv_headers.hpp"

namespace boost { namespace python {

namespace aux
{
    struct array_object_manager_traits
    {
        static bool check(PyObject* obj);
        static detail::new_non_null_reference adopt(PyObject* obj);
        static PyTypeObject const* get_pytype() ;
    };
} // namespace aux

struct ndarray : object
{
public:
    ndarray() : object() {}
    template <class T>
    explicit ndarray(T const& x) : object(x) {}

    int ndim() const;
    const Py_intptr_t *shape() const;
    const Py_intptr_t *strides() const;
    int itemsize() const;
    int dtype() const;
    const void *data() const;
    const void *getptr1(int i1) const;
    const void *getptr2(int i1, int i2) const;
    const void *getptr3(int i1, int i2, int i3) const;
    bool iscontiguous() const;
    
    bool last_dim_as_cvchannel() const;
    int cvrank() const; // = ndim() - last_dim_as_cvchannel()
    
public: // implementation detail - do not touch.
    BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(ndarray, object);
};


namespace converter
{
  template <>
  struct object_manager_traits< ndarray >
      : aux::array_object_manager_traits
  {
      BOOST_STATIC_CONSTANT(bool, is_specialized = true);
  };
}



ndarray simplenew(int len, const int *shape, int dtype);

// be careful with the new_() function. you must keep the data alive until the ndarray is deleted.
ndarray new_(int len, const int *shape, int dtype, const int *strides, void *data, int flags);


// ================================================================================================

// dtypeof
template<typename T>
int dtypeof()
{
    char s[300];
    sprintf( s, "Instantiation of function dtypeof() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

template<> int dtypeof<char>();
template<> int dtypeof<unsigned char>();
template<> int dtypeof<short>();
template<> int dtypeof<unsigned short>();
template<> int dtypeof<long>();
template<> int dtypeof<unsigned long>();
template<> int dtypeof<int>();
template<> int dtypeof<unsigned int>();
template<> int dtypeof<float>();
template<> int dtypeof<double>();

// ================================================================================================

// convert_ndarray
template<typename T>
void convert_ndarray( const ndarray &in_arr, T &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function convert_ndarray() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

template<typename T>
void convert_ndarray( const T &in_arr, ndarray &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function convert_ndarray() for class '%s' is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

// ================================================================================================

// ndarray_to_vector, convert from an ndarray to a std::vector of fixed-size elements
template<typename T>
void ndarray_to_vector( const ndarray &in_arr, std::vector<T> &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function ndarray_to_vector() for class std::vector< '%s' > is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

#define NDARRAY_TO_VECTOR(Type) template<> void ndarray_to_vector<Type>( const ndarray &in_arr, std::vector<Type> &out_arr )

// basic
NDARRAY_TO_VECTOR(char);
NDARRAY_TO_VECTOR(unsigned char);
NDARRAY_TO_VECTOR(short);
NDARRAY_TO_VECTOR(unsigned short);
NDARRAY_TO_VECTOR(long);
NDARRAY_TO_VECTOR(unsigned long);
NDARRAY_TO_VECTOR(int);
NDARRAY_TO_VECTOR(unsigned int);
NDARRAY_TO_VECTOR(float);
NDARRAY_TO_VECTOR(double);

// Vec-like
NDARRAY_TO_VECTOR(cv::Vec2b);
NDARRAY_TO_VECTOR(cv::Vec3b);
NDARRAY_TO_VECTOR(cv::Vec4b);
NDARRAY_TO_VECTOR(cv::Vec2s);
NDARRAY_TO_VECTOR(cv::Vec3s);
NDARRAY_TO_VECTOR(cv::Vec4s);
NDARRAY_TO_VECTOR(cv::Vec2w);
NDARRAY_TO_VECTOR(cv::Vec3w);
NDARRAY_TO_VECTOR(cv::Vec4w);
NDARRAY_TO_VECTOR(cv::Vec2i);
NDARRAY_TO_VECTOR(cv::Vec3i);
NDARRAY_TO_VECTOR(cv::Vec4i);
NDARRAY_TO_VECTOR(cv::Vec2f);
NDARRAY_TO_VECTOR(cv::Vec3f);
NDARRAY_TO_VECTOR(cv::Vec4f);
NDARRAY_TO_VECTOR(cv::Vec6f);
NDARRAY_TO_VECTOR(cv::Vec2d);
NDARRAY_TO_VECTOR(cv::Vec3d);
NDARRAY_TO_VECTOR(cv::Vec4d);
NDARRAY_TO_VECTOR(cv::Vec6d);

// Point-like
NDARRAY_TO_VECTOR(cv::Point2i);
NDARRAY_TO_VECTOR(cv::Point2f);
NDARRAY_TO_VECTOR(cv::Point2d);
NDARRAY_TO_VECTOR(cv::Point3i);
NDARRAY_TO_VECTOR(cv::Point3f);
NDARRAY_TO_VECTOR(cv::Point3d);

// Rect-like
NDARRAY_TO_VECTOR(cv::Rect);
NDARRAY_TO_VECTOR(cv::Rectf);
NDARRAY_TO_VECTOR(cv::Rectd);
NDARRAY_TO_VECTOR(cv::RotatedRect);

// Size-like
NDARRAY_TO_VECTOR(cv::Size2i);
NDARRAY_TO_VECTOR(cv::Size2f);
NDARRAY_TO_VECTOR(cv::Size2d);

// Scalar
NDARRAY_TO_VECTOR(cv::Scalar);

// Range
NDARRAY_TO_VECTOR(cv::Range);

// ================================================================================================

// vector_to_ndarray, convert from a std::vector of fixed-size elements to an ndarray
template<typename T>
void vector_to_ndarray( const std::vector<T> &in_arr, ndarray &out_arr )
{
    char s[300];
    sprintf( s, "Instantiation of function vector_to_ndarray() for class std::vector< '%s' > is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

#define VECTOR_TO_NDARRAY(Type) template<> void vector_to_ndarray<Type>( const std::vector<Type> &in_arr, ndarray &out_arr )

// basic
VECTOR_TO_NDARRAY(char);
VECTOR_TO_NDARRAY(unsigned char);
VECTOR_TO_NDARRAY(short);
VECTOR_TO_NDARRAY(unsigned short);
VECTOR_TO_NDARRAY(long);
VECTOR_TO_NDARRAY(unsigned long);
VECTOR_TO_NDARRAY(int);
VECTOR_TO_NDARRAY(unsigned int);
VECTOR_TO_NDARRAY(float);
VECTOR_TO_NDARRAY(double);

// Vec-like
VECTOR_TO_NDARRAY(cv::Vec2b);
VECTOR_TO_NDARRAY(cv::Vec3b);
VECTOR_TO_NDARRAY(cv::Vec4b);
VECTOR_TO_NDARRAY(cv::Vec2s);
VECTOR_TO_NDARRAY(cv::Vec3s);
VECTOR_TO_NDARRAY(cv::Vec4s);
VECTOR_TO_NDARRAY(cv::Vec2w);
VECTOR_TO_NDARRAY(cv::Vec3w);
VECTOR_TO_NDARRAY(cv::Vec4w);
VECTOR_TO_NDARRAY(cv::Vec2i);
VECTOR_TO_NDARRAY(cv::Vec3i);
VECTOR_TO_NDARRAY(cv::Vec4i);
VECTOR_TO_NDARRAY(cv::Vec2f);
VECTOR_TO_NDARRAY(cv::Vec3f);
VECTOR_TO_NDARRAY(cv::Vec4f);
VECTOR_TO_NDARRAY(cv::Vec6f);
VECTOR_TO_NDARRAY(cv::Vec2d);
VECTOR_TO_NDARRAY(cv::Vec3d);
VECTOR_TO_NDARRAY(cv::Vec4d);
VECTOR_TO_NDARRAY(cv::Vec6d);

// Point-like
VECTOR_TO_NDARRAY(cv::Point2i);
VECTOR_TO_NDARRAY(cv::Point2f);
VECTOR_TO_NDARRAY(cv::Point2d);
VECTOR_TO_NDARRAY(cv::Point3i);
VECTOR_TO_NDARRAY(cv::Point3f);
VECTOR_TO_NDARRAY(cv::Point3d);

// Rect-like
VECTOR_TO_NDARRAY(cv::Rect);
VECTOR_TO_NDARRAY(cv::Rectf);
VECTOR_TO_NDARRAY(cv::Rectd);
VECTOR_TO_NDARRAY(cv::RotatedRect);

// Size-like
VECTOR_TO_NDARRAY(cv::Size2i);
VECTOR_TO_NDARRAY(cv::Size2f);
VECTOR_TO_NDARRAY(cv::Size2d);

// Scalar
VECTOR_TO_NDARRAY(cv::Scalar);

// Range
VECTOR_TO_NDARRAY(cv::Range);


// ================================================================================================

// as_ndarray -- convert but share data
template<typename T>
ndarray as_ndarray(const object &obj)
{
    char s[300];
    sprintf( s, "Instantiation of function as_ndarray< cv::%s >() is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

#define AS_NDARRAY(Type) template<> ndarray as_ndarray<Type>(const object &obj)

// Vec-like
AS_NDARRAY(cv::Vec2b);
AS_NDARRAY(cv::Vec3b);
AS_NDARRAY(cv::Vec4b);
AS_NDARRAY(cv::Vec2s);
AS_NDARRAY(cv::Vec3s);
AS_NDARRAY(cv::Vec4s);
AS_NDARRAY(cv::Vec2w);
AS_NDARRAY(cv::Vec3w);
AS_NDARRAY(cv::Vec4w);
AS_NDARRAY(cv::Vec2i);
AS_NDARRAY(cv::Vec3i);
AS_NDARRAY(cv::Vec4i);
AS_NDARRAY(cv::Vec2f);
AS_NDARRAY(cv::Vec3f);
AS_NDARRAY(cv::Vec4f);
AS_NDARRAY(cv::Vec6f);
AS_NDARRAY(cv::Vec2d);
AS_NDARRAY(cv::Vec3d);
AS_NDARRAY(cv::Vec4d);
AS_NDARRAY(cv::Vec6d);

// Point-like
AS_NDARRAY(cv::Point2i);
AS_NDARRAY(cv::Point2f);
AS_NDARRAY(cv::Point2d);
AS_NDARRAY(cv::Point3i);
AS_NDARRAY(cv::Point3f);
AS_NDARRAY(cv::Point3d);

// Rect-like
AS_NDARRAY(cv::Rect);
AS_NDARRAY(cv::Rectf);
AS_NDARRAY(cv::Rectd);
AS_NDARRAY(cv::RotatedRect);

// Size-like
AS_NDARRAY(cv::Size2i);
AS_NDARRAY(cv::Size2f);
AS_NDARRAY(cv::Size2d);

// Scalar
AS_NDARRAY(cv::Scalar);

// Range
AS_NDARRAY(cv::Range);

// Mat
AS_NDARRAY(cv::Mat);

// MatND
AS_NDARRAY(cv::MatND);

// ================================================================================================

// from_ndarray -- convert but share data
template<typename T>
object from_ndarray(const ndarray &arr)
{
    char s[300];
    sprintf( s, "Instantiation of function from_ndarray< cv::%s >() is not yet implemented.", typeid(T).name() );
    PyErr_SetString(PyExc_NotImplementedError, s);
    throw error_already_set(); 
}

#define FROM_NDARRAY(Type) template<> object from_ndarray<Type>(const ndarray &arr)

// Vec-like
FROM_NDARRAY(cv::Vec2b);
FROM_NDARRAY(cv::Vec3b);
FROM_NDARRAY(cv::Vec4b);
FROM_NDARRAY(cv::Vec2s);
FROM_NDARRAY(cv::Vec3s);
FROM_NDARRAY(cv::Vec4s);
FROM_NDARRAY(cv::Vec2w);
FROM_NDARRAY(cv::Vec3w);
FROM_NDARRAY(cv::Vec4w);
FROM_NDARRAY(cv::Vec2i);
FROM_NDARRAY(cv::Vec3i);
FROM_NDARRAY(cv::Vec4i);
FROM_NDARRAY(cv::Vec2f);
FROM_NDARRAY(cv::Vec3f);
FROM_NDARRAY(cv::Vec4f);
FROM_NDARRAY(cv::Vec6f);
FROM_NDARRAY(cv::Vec2d);
FROM_NDARRAY(cv::Vec3d);
FROM_NDARRAY(cv::Vec4d);
FROM_NDARRAY(cv::Vec6d);

// Point-like
FROM_NDARRAY(cv::Point2i);
FROM_NDARRAY(cv::Point2f);
FROM_NDARRAY(cv::Point2d);
FROM_NDARRAY(cv::Point3i);
FROM_NDARRAY(cv::Point3f);
FROM_NDARRAY(cv::Point3d);

// Rect-like
FROM_NDARRAY(cv::Rect);
FROM_NDARRAY(cv::Rectf);
FROM_NDARRAY(cv::Rectd);
FROM_NDARRAY(cv::RotatedRect);

// Size-like
FROM_NDARRAY(cv::Size2i);
FROM_NDARRAY(cv::Size2f);
FROM_NDARRAY(cv::Size2d);

// Scalar
FROM_NDARRAY(cv::Scalar);

// Range
FROM_NDARRAY(cv::Range);

// Mat
FROM_NDARRAY(cv::Mat);

// MatND
FROM_NDARRAY(cv::MatND);


}} // namespace boost::python

#endif // SD_NDARRAY_HPP
