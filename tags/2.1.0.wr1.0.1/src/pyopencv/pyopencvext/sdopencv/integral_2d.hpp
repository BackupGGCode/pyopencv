#ifndef SDOPENCV_INTERGAL_IMAGE_2D_HPP
#define SDOPENCV_INTERGAL_IMAGE_2D_HPP

#include <cxcore.h>

namespace sdopencv
{
    // Consult cv::integral() for more details about the parameters
    class IntegralImage
    {    
        public:
            int sdepth;
            
            cv::Mat integral;            
            cv::Size image_size;
            
            // same as sum() below, but without checking
            template<typename T>
            cv::Scalar sum_rect(const cv::Rect &rect)
            {
                cv::Scalar s;
                int nc = integral.channels();
                int x = rect.width*nc;
                T *d1 = (T *)(integral.ptr(rect.y))+rect.x*nc; // top-left
                T *d2 = d1 + rect.height*integral.step/sizeof(T); // bottom-left
                while(--nc >= 0) s.val[nc] = d1[nc]+d2[x+nc]-d1[x+nc]-d2[nc];
                return s;
            }
            
            // adjust a rectangle w.r.t. to the image size
            // return true if the rectangle has at least 1 pixel
            bool adjust_rect(cv::Rect &rect);
            
            // sdepth=CV_32S, CV_32F, or CV_64F only
            IntegralImage(int sdepth=CV_64F);

            // input image
            void operator()(const cv::Mat &image);

            // compute the sum of intensities of pixels inside a cv::Rect
            // here, a pixel pt is inside rect if
            //   rect.x <= pt.x < rect.x+rect.width, and
            //   rect.y <= pt.y < rect.y+rect.height
            // note that if there is no pixel satisfying these conditions, zero is returned
            cv::Scalar sum(cv::Rect rect);
    };
    
    void patchBasedStdDev(cv::Mat const &in_image, cv::Size const &patch_size, cv::Mat &out_image);
}

#endif // SDCPP_DIFFERENTIALCHARACTERISTICS_H
