#include <boost/python/detail/prefix.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/object.hpp>

#include <boost/mpl/equal.hpp>

#include <algorithm>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>

#include "opencv_converters.hpp"


// ================================================================================================

// workaround for getting a CvMat pointer
CvMat * get_CvMat_ptr(cv::Mat const &mat)
{
    if(mat.empty()) return 0;
    static int cnt = 0;
    static CvMat arr[1024];
    CvMat *result = &(arr[cnt] = mat);
    cnt = (cnt+1) & 1023;
    return result;
}


// workaround for getting an IplImage pointer
IplImage * get_IplImage_ptr(cv::Mat const &mat)
{
    if(mat.empty()) return 0;
    static int cnt = 0;
    static IplImage arr[1024];
    IplImage *result = &(arr[cnt] = mat);
    cnt = (cnt+1) & 1023;
    return result;
}

// ------------------------------------------------------------------------------------------------
// convert from a sequence of Mat to vector of Mat-equivalent type
// i.e. IplImage, CvMat, IplImage *, CvMat *, cv::Mat, cv::Mat *

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(IplImage)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = (cv::Mat const &)(bp::extract<cv::Mat const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(IplImage *)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = get_IplImage_ptr(bp::extract<cv::Mat const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(CvMat)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = (cv::Mat const &)(bp::extract<cv::Mat const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(CvMat *)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = get_CvMat_ptr(bp::extract<cv::Mat const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Mat)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = bp::extract<cv::Mat &>(in_arr[i]);
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Mat *)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = bp::extract<cv::Mat *>(in_arr[i]);
}

CONVERT_FROM_SEQ_OF_MAT_TO_VECTOR_OF_T(cv::Ptr<cv::Mat>)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i)
    {
        cv::Mat *obj = new cv::Mat();
        *obj = bp::extract<cv::Mat const &>(in_arr[i]);
        out_arr[i] = cv::Ptr<cv::Mat>(obj);
    }
}



// ================================================================================================

// workaround for getting a CvMatND pointer
CvMatND * get_CvMatND_ptr(cv::MatND const &matnd)
{
    static int cnt = 0;
    static CvMatND arr[1024];
    CvMatND *result = &(arr[cnt] = matnd);
    cnt = (cnt+1) & 1023;
    return result;
}


// convert from a sequence of MatND to vector of MatND-equivalent type
// i.e. CvMatND, CvMatND *, cv::MatND, cv::MatND *
CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(CvMatND)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = (cv::MatND const &)(bp::extract<cv::MatND const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(CvMatND *)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = get_CvMatND_ptr(bp::extract<cv::MatND const &>(in_arr[i]));
}

CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(cv::MatND)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = bp::extract<cv::MatND &>(in_arr[i]);
}

CONVERT_FROM_SEQ_OF_MATND_TO_VECTOR_OF_T(cv::MatND *)
{
    bp::object const &in_arr = in_seq.get_obj();
    int len = bp::len(in_arr);
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) 
        out_arr[i] = bp::extract<cv::MatND *>(in_arr[i]);
}




// ================================================================================================

// convert_vector_to_seq

#define CONVERT_VECTOR_TO_NDARRAY(VectType) \
CONVERT_VECTOR_TO_SEQ(VectType) \
{ \
    sdcpp::ndarray out_arr = sdcpp::simplenew_ndarray(0,0,5); \
    sdcpp::vector_to_ndarray(in_arr, out_arr); \
    return sdcpp::sequence(out_arr.get_obj()); \
}

// basic
CONVERT_VECTOR_TO_NDARRAY(char);
CONVERT_VECTOR_TO_NDARRAY(unsigned char);
CONVERT_VECTOR_TO_NDARRAY(short);
CONVERT_VECTOR_TO_NDARRAY(unsigned short);
CONVERT_VECTOR_TO_NDARRAY(long);
CONVERT_VECTOR_TO_NDARRAY(unsigned long);
CONVERT_VECTOR_TO_NDARRAY(int);
CONVERT_VECTOR_TO_NDARRAY(unsigned int);
CONVERT_VECTOR_TO_NDARRAY(float);
CONVERT_VECTOR_TO_NDARRAY(double);

// Vec-like
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2b);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3b);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4b);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2s);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3s);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4s);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2w);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3w);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4w);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2i);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3i);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4i);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2f);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3f);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4f);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec6f);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec2d);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec3d);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec4d);
CONVERT_VECTOR_TO_NDARRAY(cv::Vec6d);

// Point-like
CONVERT_VECTOR_TO_NDARRAY(cv::Point2i);
CONVERT_VECTOR_TO_NDARRAY(cv::Point2f);
CONVERT_VECTOR_TO_NDARRAY(cv::Point2d);
CONVERT_VECTOR_TO_NDARRAY(cv::Point3i);
CONVERT_VECTOR_TO_NDARRAY(cv::Point3f);
CONVERT_VECTOR_TO_NDARRAY(cv::Point3d);

// Rect-like
CONVERT_VECTOR_TO_NDARRAY(cv::Rect);
CONVERT_VECTOR_TO_NDARRAY(cv::Rectf);
CONVERT_VECTOR_TO_NDARRAY(cv::Rectd);
CONVERT_VECTOR_TO_NDARRAY(cv::RotatedRect);

// Size-like
CONVERT_VECTOR_TO_NDARRAY(cv::Size2i);
CONVERT_VECTOR_TO_NDARRAY(cv::Size2f);
CONVERT_VECTOR_TO_NDARRAY(cv::Size2d);

// Scalar
CONVERT_VECTOR_TO_NDARRAY(cv::Scalar);

// Range
CONVERT_VECTOR_TO_NDARRAY(cv::Range);



// ================================================================================================

namespace mpl=boost::mpl;

template<typename T>
struct sdconverter;

template<typename T>
struct sdconverter_simple
{
    typedef T pytype;
    void from_python(bp::object const &py_obj, T &cpp_obj) { cpp_obj = bp::extract<T>(py_obj); }
    void to_python(T const &cpp_obj, bp::object &py_obj)
    {
        T &py_obj2 = bp::extract<T &>(py_obj);
        py_obj2 = cpp_obj;
    }
    T new_pyobj() { return T(); }
};

template<>
struct sdconverter_simple< cv::Ptr<cv::Mat> >
{
    typedef cv::Mat pytype;
    void from_python(bp::object const &py_obj, cv::Ptr<cv::Mat> &cpp_obj)
    {
        cv::Mat *py_obj2 = new cv::Mat();
        *py_obj2 = bp::extract<cv::Mat const &>(py_obj);
        cpp_obj = cv::Ptr<cv::Mat>(py_obj2);
    }
    void to_python(cv::Ptr<cv::Mat> const &cpp_obj, bp::object &py_obj)
    {
        pytype &py_obj2 = bp::extract<pytype &>(py_obj);
        py_obj2 = *cpp_obj;
    }
    cv::Ptr<cv::Mat> new_pyobj() { return cv::Ptr<cv::Mat>(new cv::Mat()); }
};

template<typename T>
struct sdconverter_vector
{
    typedef bp::list pytype;
    void from_python(bp::object const &py_obj, std::vector<T> &cpp_obj)
    {
        int n = bp::len(py_obj);
        cpp_obj.resize(n);
        for(int i = 0; i < n; ++i)
            sdconverter<T>::impl_t::from_python(py_obj[i], cpp_obj[i]);
    }
    void to_python(std::vector<T> const &cpp_obj, bp::object &py_obj)
    {
        int n = cpp_obj.size(), n2 = bp::len(py_obj);
        bp::extract<bp::list> py_obj2(py_obj);
        while(n2 > n) { py_obj2().pop(); --n2; }
        if(!n) return;
        bp::object obj;
        for(int i = 0; i < n; ++i)
        {
            if(i == n2)
            {
                py_obj2().append(bp::object(sdconverter<T>::impl_t::new_pyobj()));
                ++n2;
            }
            else if(!bp::extract<typename sdconverter<T>::impl_t::pytype>(py_obj[i]).check())
                py_obj[i] = bp::object(sdconverter<T>::impl_t::new_pyobj());
                
            obj = py_obj[i];
            sdconverter<T>::impl_t::to_python(cpp_obj[i], obj);
            py_obj[i] = obj;
        }
    }
    bp::list new_pyobj() { return bp::list(); }
};

template<typename T>
struct sdconverter
{
    typedef typename mpl::if_<
        mpl::equal< std::vector<typename mpl::deref<typename mpl::begin<T>::type >::type >, T >,
        sdconverter_vector<typename mpl::deref<typename mpl::begin<T>::type >::type >,
        sdconverter_simple<T>
    >::type impl_t;
};