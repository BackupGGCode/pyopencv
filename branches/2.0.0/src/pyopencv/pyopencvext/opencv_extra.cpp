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
#include "opencv_converters.hpp"
#include "ndarray.hpp"


// ================================================================================================

void CV_CDECL sdTrackbarCallback2(int pos, void* userdata)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)userdata)));
    if(bp::object(items[0]).ptr() != Py_None) // invoke if not None
        (items[0])(pos, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}


void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)param)));
    if(bp::object(items[0]).ptr() != Py_None) // invoke if not None
        (items[0])(event, x, y, flags, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}

float CV_CDECL sdDistanceFunction( const float* a, const float*b, void* user_param )
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)user_param)));
    // pass 'a' and 'b' by address instead of by pointer
    return bp::extract < float >((items[0])((int)a, (int)b, bp::object(items[1]))); // need a copy of items[1] to make it safe with threading
}

// ================================================================================================

bp::sequence mixChannels(const bp::sequence &src, bp::sequence &dst, const bp::ndarray &fromTo)
{
    char s[200];
    
    const Py_intptr_t *shape = fromTo.shape();
    
    if(fromTo.ndim() != 2 || fromTo.dtype() != bp::dtypeof<long>() || shape[1] != 2 || !fromTo.iscontiguous())
    {
        sprintf(s, "Wrong type! 'fromTo' is not a N-row 2-column int32 C-contiguous ndarray. ");
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }
    
    bp::extract<const cv::Mat &> mat(src[0]);
    if(mat.check())
    {
        std::vector<cv::Mat> src2, dst2;
        convert_seq_to_vector(src, src2);
        convert_seq_to_vector(dst, dst2);
        cv::mixChannels(&src2[0], bp::len(src), &dst2[0], bp::len(dst), (const int *)fromTo.data(), shape[0]);
        return convert_vector_to_seq(dst2);
    }

    bp::extract<const cv::MatND &> matnd(src[0]);
    if(matnd.check())
    {
        std::vector<cv::MatND> src3, dst3;
        convert_seq_to_vector(src, src3);
        convert_seq_to_vector(dst, dst3);
        cv::mixChannels(&src3[0], bp::len(src), &dst3[0], bp::len(dst), (const int *)fromTo.data(), shape[0]);
        return convert_vector_to_seq(dst3);
    }

    sprintf(s, "Cannot determine whether the 1st item of 'src' is Mat or MatND.");
    PyErr_SetString(PyExc_TypeError, s);
    throw bp::error_already_set();
    
    return bp::sequence();
}

bp::tuple minMaxLoc(const bp::object& a, const bp::object& mask)
{
    double minVal, maxVal;
    int minIdx[CV_MAX_DIM], maxIdx[CV_MAX_DIM];
    int i, n;
    cv::Point minLoc, maxLoc;
    
    bp::tuple result;
    
    bp::extract<const cv::Mat &> mat(a);
    bp::extract<const cv::MatND &> matnd(a);
    bp::extract<const cv::SparseMat &> smat(a);
    if(mat.check())
    {
        if(mask.ptr() == Py_None) // None object
            cv::minMaxLoc(mat(), &minVal, &maxVal, &minLoc, &maxLoc);
        else
            cv::minMaxLoc(mat(), &minVal, &maxVal, &minLoc, &maxLoc, bp::extract<const cv::Mat &>(mask));
        result = bp::make_tuple(bp::object(minVal), bp::object(maxVal), bp::object(minLoc), bp::object(maxLoc));
    }
    else if(matnd.check())
    {
        const cv::MatND &m = matnd();
        if(mask.ptr() == Py_None) // None object
            cv::minMaxLoc(m, &minVal, &maxVal, minIdx, maxIdx);
        else
            cv::minMaxLoc(m, &minVal, &maxVal, minIdx, maxIdx, bp::extract<const cv::MatND &>(mask));
        n = m.dims;
        bp::list l1, l2;
        for(i = 0; i < n; ++i)
        {
            l1.append(bp::object(minIdx[i]));
            l2.append(bp::object(maxIdx[i]));
        }
        result = bp::make_tuple(bp::object(minVal), bp::object(maxVal), bp::tuple(l1), bp::tuple(l2));
    }
    else if(smat.check())
    {
        const cv::SparseMat &m2 = smat();
        cv::minMaxLoc(m2, &minVal, &maxVal, minIdx, maxIdx);
        n = m2.dims();
        bp::list l1, l2;
        for(i = 0; i < n; ++i)
        {
            l1.append(bp::object(minIdx[i]));
            l2.append(bp::object(maxIdx[i]));
        }
        result = bp::make_tuple(bp::object(minVal), bp::object(maxVal), bp::tuple(l1), bp::tuple(l2));
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Cannot determine whether 'a' is Mat, MatND, or SparseMat.");
        throw bp::error_already_set();
    }
    return result;
}


// ================================================================================================

