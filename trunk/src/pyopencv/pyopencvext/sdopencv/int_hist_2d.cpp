#include <algorithm>
#include <cv.h>

#include "dtype.hpp"
#include "int_hist_2d.hpp"

using namespace std;

namespace sdopencv
{
    IntegralHistogram::IntegralHistogram(int _histSize, const std::vector<float> &_ranges, bool _uniform)
        : histSize(_histSize), ranges(_ranges), uniform(_uniform)
    {
        n_iimages = (histSize-1)/4+1;
        last_nc = ((histSize-1)&3)+1;
        iimages.resize(n_iimages);
        for(int i = 0; i < n_iimages; ++i) iimages[i].sdepth = CV_32S;
        if(!uniform) ranges.resize(histSize); // truncate unimportant elements of ranges
    }
    
    int IntegralHistogram::get_index(float value)
    {
        if(value < ranges[0]) return -1;
        if(uniform)
        {
            if(value >= ranges[1]) return histSize;
            return (int)floor((value-ranges[0])*histSize/(ranges[1]-ranges[0]));
        }
        if(value >= ranges[histSize-1]) return histSize;
        return &(*lower_bound(ranges.begin(), ranges.end(), value))-&ranges[1];
    }
    
            
    void IntegralHistogram::operator()(const cv::Mat &image)
    {
        if(image.channels() > 1) CV_Error( -1, "Unsupported multi-channel images" );

        // allocate a few binary images
        vector<cv::Mat> bin_images; bin_images.resize(n_iimages);
        int i, y, x;
        
        // clear the binary images
        for(i = 0; i < n_iimages-1; ++i) bin_images[i].create(image.size(), CV_8UC4);
        bin_images[i].create(image.size(), CV_MAKETYPE(CV_8U, last_nc));
        for(i = 0; i < n_iimages; ++i) bin_images[i].setTo(cv::Scalar());
        
        
        // populate the input image' pixels to the binary images
        for(y = 0; y < image.rows; ++y)
        {
            // get the indices per row
            vector<int> indices;
            switch(image.depth())
            {
            case CV_8U: values_to_bin_indices(image.ptr<uint8>(y), image.cols, indices); break;
            case CV_8S: values_to_bin_indices(image.ptr<int8>(y), image.cols, indices); break;
            case CV_16S: values_to_bin_indices(image.ptr<int16>(y), image.cols, indices); break;
            case CV_32S: values_to_bin_indices(image.ptr<int>(y), image.cols, indices); break;
            case CV_32F: values_to_bin_indices(image.ptr<float32>(y), image.cols, indices); break;
            case CV_64F: values_to_bin_indices(image.ptr<float64>(y), image.cols, indices); break;
            default: CV_Error( -1, "Unsupported image depth" );
            }
            
            // populate them
            int v1, v2;
            for(x = 0; x < image.cols; ++x)
            {
                v1 = indices[x];
                if(v1 < 0 || v1 >= histSize) continue;
                v2 = v1&3; v1 >>= 2;
                *(bin_images[v1].ptr(y) + bin_images[v1].channels()*x + v2) = 1;
            }
        }
        
        // feed the binary images to the integral images
        for(i = 0; i < n_iimages; ++i) iimages[i](bin_images[i]);        
    }
    
    
    void IntegralHistogram::calcHist(cv::Rect rect, std::vector<int> &out_hist)
    {
        int i;
        int *dhist;
        out_hist.resize(histSize);
        if(iimages[0].adjust_rect(rect))
        {
            cv::Scalar s;
            for(i = 0, dhist = &out_hist[0]; i < n_iimages-1; ++i, dhist += 4)
            {
                s = iimages[i].sum_rect<int>(rect);
                dhist[0] = (int) s.val[0];
                dhist[1] = (int) s.val[1];
                dhist[2] = (int) s.val[2];
                dhist[3] = (int) s.val[3];
            }
            
            s = iimages[i].sum_rect<int>(rect);
            i = last_nc; while(--i >= 0) dhist[i] = (int) s.val[i];
        }
        else
            for(i = 0; i < histSize; ++i) out_hist[i] = 0;
    }
}
