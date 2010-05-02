#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include "iterators.hpp"

using namespace std;

namespace bp = boost::python;

namespace sdopencv
{


LineIterator::LineIterator(const cv::Mat& img, cv::Point const &pt1, 
    cv::Point const &pt2, int connectivity, bool leftToRight)
    : cv::LineIterator(img, pt1, pt2, connectivity, leftToRight)
{
    iteration = 0;
    ptr0 = img.data;
    ws = img.step;
    es = img.elemSize();
}

cv::Point LineIterator::next()
{
    int ofs = (int)(ptr-ptr0);
    
    if(iteration < count)
    {
        ++(*this);
        ++iteration;
    }
    else
    {
        PyErr_SetString(PyExc_StopIteration, "No more pixel.");
        throw bp::error_already_set(); 
    }
    
    return cv::Point((ofs%ws)/es, ofs/ws);
}


}
