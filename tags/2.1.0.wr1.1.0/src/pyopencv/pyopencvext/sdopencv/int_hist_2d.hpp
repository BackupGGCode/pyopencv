#ifndef SDOPENCV_INTERGAL_HISTOGRAM_2D_HPP
#define SDOPENCV_INTERGAL_HISTOGRAM_2D_HPP

#include <cxcore.h>
#include <vector>
#include "integral_2d.hpp"

namespace sdopencv
{
    // Consult cv::calcHist() for more details about the parameters
    // Here, we deal with single-channel only, so dims=1.
    // Thus, histSize is an integer instead of a 1D vector.
    // And ranges is a 1D vector instead of a 2D vector.
    class IntegralHistogram
    {
        protected:
            int histSize;
            std::vector<float> ranges;
            bool uniform;
            
            std::vector<IntegralImage> iimages;
            int n_iimages, last_nc;
            
        public:
            IntegralHistogram(int histSize, const std::vector<float> &ranges, bool uniform=true);

            // single-channel input image
            void operator()(const cv::Mat &image);

            // compute the histogram of intensities of pixels inside a cv::Rect
            // here, a pixel pt is inside rect if
            //   rect.x <= pt.x < rect.x+rect.width, and
            //   rect.y <= pt.y < rect.y+rect.height
            // note that if there is no pixel satisfying these conditions, a vector of zeros is returned
            void calcHist(cv::Rect rect, std::vector<int> &out_hist);
            
            // get bin index from value
            int get_index(float value);
            
            template<typename T>
            void values_to_bin_indices(T *values, int n, std::vector<int> &indices)
            {
                indices.resize(n);
                while(--n >= 0) indices[n] = get_index((float)values[n]);
            }
    };
}

#endif // SDOPENCV_INTERGAL_HISTOGRAM_2D_HPP
