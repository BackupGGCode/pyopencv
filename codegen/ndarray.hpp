// Copyright David Abrahams 2002.
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

    void check() const;

    int ndim() const;
    const int *shape() const;
    const int *strides() const;
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
    const char message[] = "Instantiation of function dtypeof() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
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
    const char message[] = "Instantiation of function convert_ndarray() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw error_already_set(); 
}

template<typename T>
void convert_ndarray( const T &in_arr, ndarray &out_arr )
{
    const char message[] = "Instantiation of function convert_ndarray() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw error_already_set(); 
}

// convert_ndarray, std::vector case
// Note: because Python and C have different ways of allocating/reallocating memory,
// it is UNSAFE to share data between ndarray and std::vector.
// In this implementation, data is allocated and copied instead.
template<typename T>
void convert_ndarray( const ndarray &in_arr, std::vector<T> &out_arr )
{
    char s[100];
    int nd = in_arr.ndim();
    if(nd != 1)
    {
        sprintf(s, "Rank must be 1, rank=%d detected.", nd);
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set(); 
    }
    if(in_arr.dtype() != dtypeof<T>())
    {
        sprintf(s, "Ndarray's element type is not the same as that of std::vector. ndarray's dtype=%d, vector's dtype=%d.", in_arr.dtype(), dtypeof<T>());
        PyErr_SetString(PyExc_TypeError, s);
        throw error_already_set(); 
    }
    
    int len = in_arr.shape()[0];
    
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = *(T *)in_arr.getptr1(i);
}

extern template void convert_ndarray( const ndarray &in_arr, std::vector<char> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned char> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<short> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned short> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<long> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned long> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<int> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<unsigned int> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<float> &out_arr );
extern template void convert_ndarray( const ndarray &in_arr, std::vector<double> &out_arr );

// convert_ndarray, std::vector case
template<typename T>
void convert_ndarray( const std::vector<T> &in_arr, ndarray &out_arr )
{
    int len = in_arr.size();
    out_arr = simplenew(1, &len, dtypeof<T>());
    T *data = (T *)out_arr.data();
    for(int i = 0; i < len; ++i) data[i] = in_arr[i];
}

extern template void convert_ndarray( const std::vector<char> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<unsigned char> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<short> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<unsigned short> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<long> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<unsigned long> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<int> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<unsigned int> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<float> &in_arr, ndarray &out_arr );
extern template void convert_ndarray( const std::vector<double> &in_arr, ndarray &out_arr );


ndarray as_ndarray(const object &obj);

object as_Scalar(const ndarray &arr);
object as_Mat(const ndarray &arr);
object as_MatND(const ndarray &arr);

void mixChannels(const tuple src, tuple dst, const ndarray &fromTo);

}} // namespace boost::python

#endif // SD_NDARRAY_HPP
