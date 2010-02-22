#include <boost/python/detail/prefix.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/object.hpp>

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


// ================================================================================================

// convert_vector_to_seq

#define CONVERT_VECTOR_TO_NDARRAY(VectType) \
CONVERT_VECTOR_TO_SEQ(VectType) \
{ \
    bp::ndarray out_arr; \
    bp::vector_to_ndarray(in_arr, out_arr); \
    return bp::sequence(out_arr); \
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
CONVERT_VECTOR_TO_NDARRAY(cv::Rectd);
CONVERT_VECTOR_TO_NDARRAY(cv::Rectf);
CONVERT_VECTOR_TO_NDARRAY(cv::RotatedRect);

// Scalar
CONVERT_VECTOR_TO_NDARRAY(cv::Scalar);

// Range
CONVERT_VECTOR_TO_NDARRAY(cv::Range);



// ================================================================================================

