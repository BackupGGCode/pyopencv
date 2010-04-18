#ifndef SDOPENCV_ITERATORS_HPP
#define SDOPENCV_ITERATORS_HPP

#include <cxcore.h>

namespace sdopencv
{
    class CV_EXPORTS LineIterator : public cv::LineIterator
    {
    public:
        LineIterator(const cv::Mat& img, cv::Point const &pt1, cv::Point const &pt2,
            int connectivity=8, bool leftToRight=false);
            
        LineIterator const &iter() { return *this; }
        cv::Point next();
        
    private:
        int iteration;
        int ws, es;
        uchar *ptr0;
    };
}

#endif // SDOPENCV_ITERATORS_HPP
