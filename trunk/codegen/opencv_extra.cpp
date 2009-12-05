#include <boost/python/detail/prefix.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/object.hpp>

#include <algorithm>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>

#include "opencv_extra.hpp"


// ================================================================================================

void CV_CDECL sdTrackbarCallback2(int pos, void* userdata)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)userdata)));
    (items[0])(pos, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}


void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)param)));
    (items[0])(event, x, y, flags, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}

float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param )
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)user_param)));
    // pass 'a' and 'b' by address instead of by pointer
    return bp::extract < float >((items[0])((int)a, (int)b, bp::object(items[1]))); // need a copy of items[1] to make it safe with threading
}

// ================================================================================================

template void convert_Mat( const cv::Mat &in_arr, std::vector<char> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned char> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<short> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned short> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<long> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned long> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<int> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<unsigned int> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<float> &out_arr );
template void convert_Mat( const cv::Mat &in_arr, std::vector<double> &out_arr );

// 1-row CV_32SC2 Mat
template<> void convert_Mat<cv::Point>( const cv::Mat &in_arr, std::vector<cv::Point> &out_arr )
{
    char s[100];
    if(in_arr.type() != CV_32SC2)
    {
        sprintf(s, "Mat must be of type CV_32SC2, type=%d detected.", in_arr.type());
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    if(in_arr.rows != 1)
    {
        sprintf(s, "Mat must be a row vector, rows=%d detected.", in_arr.rows);
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set(); 
    }
    out_arr.resize(in_arr.cols);
    int *data = (int *)in_arr.data;
    for(int i = 0; i < in_arr.cols; ++i)
    {
        out_arr[i] = cv::Point(data[0], data[1]);
        data += 2;
    }
}


// ================================================================================================

template void convert_Mat( const std::vector<char> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<unsigned char> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<short> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<unsigned short> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<long> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<unsigned long> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<int> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<unsigned int> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<float> &in_arr, cv::Mat &out_arr );
template void convert_Mat( const std::vector<double> &in_arr, cv::Mat &out_arr );

// 1-row CV_32SC2 Mat
template<> void convert_Mat<cv::Point>( const std::vector<cv::Point> &in_arr, cv::Mat &out_arr )
{
    int len = in_arr.size();
    out_arr.create(cv::Size(1, len), CV_32SC2);
    int *data = (int *)out_arr.data;
    for(int i = 0; i < len; ++i)
    {
        data[0] = in_arr[i].x;
        data[1] = in_arr[i].y;
        data += 2;
    }
}


// ================================================================================================

template void convert_Mat( const cv::Mat &in_arr, char *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, unsigned char *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, short *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, unsigned short *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, long *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, unsigned long *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, int *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, unsigned int *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, float *&out_arr );
template void convert_Mat( const cv::Mat &in_arr, double *&out_arr );

// ================================================================================================

