#ifndef SDOPENCV_EXTRA_H
#define SDOPENCV_EXTRA_H

#include <vector>

#include "boost/python.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "boost/python/tuple.hpp"
#include "boost/python/to_python_value.hpp"

#include "opencv_headers.hpp"

namespace bp = boost::python;

CV_INLINE CvPyramid sdCreatePyramid( const CvArr* img, int extra_layers, double rate,
                                const CvSize* layer_sizes CV_DEFAULT(0),
                                CvArr* bufarr CV_DEFAULT(0),
                                int calc CV_DEFAULT(1),
                                int filter CV_DEFAULT(CV_GAUSSIAN_5x5) )
{
    CvPyramid pyr;
    pyr.pyramid = cvCreatePyramid(img, extra_layers, rate, layer_sizes, bufarr, calc, filter);
    pyr.extra_layers = extra_layers;
    return pyr;
}


void CV_CDECL sdTrackbarCallback2(int pos, void* userdata);
void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param);
float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param );

// ================================================================================================

// cvtypeof
template<typename T>
inline int cvtypeof()
{
    const char message[] = "Instantiation of function cvtypeof() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<> inline int cvtypeof<char>() { return CV_8S; }
template<> inline int cvtypeof<unsigned char>() { return CV_8U; }
template<> inline int cvtypeof<short>() { return CV_16S; }
template<> inline int cvtypeof<unsigned short>() { return CV_16U; }
template<> inline int cvtypeof<long>() { return CV_32S; }
template<> inline int cvtypeof<int>() { return CV_32S; }
template<> inline int cvtypeof<float>() { return CV_32F; }
template<> inline int cvtypeof<double>() { return CV_64F; }

// ================================================================================================

// convert_Mat
template<typename T>
void convert_Mat( const cv::Mat &in_arr, T &out_arr )
{
    const char message[] = "Instantiation of function convert_Mat() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

template<typename T>
void convert_Mat( const T &in_arr, cv::Mat &out_arr )
{
    const char message[] = "Instantiation of function convert_Mat() for the given class is not yet implemented.";
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw bp::error_already_set(); 
}

// convert_Mat, std::vector case
// it is UNSAFE to share data between cv::Mat and std::vector because both can resize their data.
// In this implementation, data is allocated and copied instead.
template<typename T>
void convert_Mat( const cv::Mat &in_arr, std::vector<T> &out_arr )
{
    char s[100];
    if(!in_arr.flags) { out_arr.clear(); return; }
    
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
        sprintf(s, "cv::Mat's element type is not the same as that of std::vector. cv::Mat's type=%d, vector's type=%d.", in_arr.type(), cvtypeof<T>());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    
    int len = in_arr.cols;
    T *data = (T *)in_arr.data;
    
    out_arr.resize(len);
    for(int i = 0; i < len; ++i) out_arr[i] = data[i];
}

extern template void convert_Mat( const cv::Mat &in_arr, std::vector<char> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned char> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<short> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned short> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<long> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned long> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<int> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned int> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<float> &out_arr );
extern template void convert_Mat( const cv::Mat &in_arr, std::vector<double> &out_arr );

// convert_Mat, std::vector case
template<typename T>
void convert_Mat( const std::vector<T> &in_arr, cv::Mat &out_arr )
{
    int len = in_arr.size();
    out_arr.create(1, len, cvtypeof<T>());
    T *data = (T *)out_arr.data;
    for(int i = 0; i < len; ++i) data[i] = in_arr[i];
}

extern template void convert_Mat( const std::vector<char> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<unsigned char> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<short> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<unsigned short> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<long> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<unsigned long> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<int> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<unsigned int> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<float> &in_arr, cv::Mat &out_arr );
extern template void convert_Mat( const std::vector<double> &in_arr, cv::Mat &out_arr );

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




#endif
