#include <cv.h>

#include "dtype.hpp"
#include "integral_2d.hpp"

using namespace std;

namespace sdopencv
{
    IntegralImage::IntegralImage(int _sdepth) : sdepth(_sdepth)
    {
    }

    // input image
    void IntegralImage::operator()(const cv::Mat &image)
    {
        image_size = image.size();
        integral.create(image_size, CV_MAKETYPE(sdepth, image.channels()));
        cv::integral(image, integral, sdepth);
    }

    
    // adjust a rectangle w.r.t. to the image size
    // return true if the rectangle has at least 1 pixel
    bool IntegralImage::adjust_rect(cv::Rect &rect)
    {
        if(rect.x < 0) { rect.width += rect.x; rect.x = 0; }
        if(rect.x >= image_size.width) return false;
        
        if(rect.y < 0) { rect.height += rect.y; rect.y = 0; }
        if(rect.y >= image_size.height) return false;
        
        if(rect.width <= 0) return false;
        if(rect.width+rect.x > image_size.width) rect.width = image_size.width-rect.x;
        
        if(rect.height <= 0) return false;
        if(rect.height+rect.y > image_size.height) rect.height = image_size.height-rect.y;
        
        return true;
    }

    
    // compute the sum of intensities of pixels inside a cv::Rect
    // here, a pixel pt is inside rect if
    //   rect.x <= pt.x < rect.x+rect.width, and
    //   rect.y <= pt.y < rect.y+rect.height
    // note that if there is no pixel satisfying these conditions, zero is returned
    cv::Scalar IntegralImage::sum(cv::Rect rect)
    {
        if(!adjust_rect(rect)) return cv::Scalar();
        
        if(sdepth == CV_32S) return sum_rect<int>(rect);
        if(sdepth == CV_32F) return sum_rect<float32>(rect);
        if(sdepth == CV_64F) return sum_rect<float64>(rect);
        
        CV_Error(-1, "Only CV_32S, CV_32F, and CV_64F are valid values for sdepth");
        return cv::Scalar();
    }
    
    void patchBasedStdDev(cv::Mat const &in_image, cv::Size const &patch_size, cv::Mat &out_image)
    {
        // preparation
        cv::Rect r(cv::Point(0, 0), patch_size);
        int h = in_image.rows - patch_size.height;
        int w = in_image.cols - patch_size.width;
        int nc = in_image.channels();
        int out_depth = CV_MAKETYPE(CV_64F, nc);
        double s = 1.0/patch_size.area();
        out_image.create(cv::Size(w, h), out_depth);
        
        // 1st-order moment
        IntegralImage ii; ii(in_image);
    
        // 2nd-order moment
        cv::Mat sqr_image(in_image.size(), out_depth);
        in_image.convertTo(sqr_image, sqr_image.depth());
        cv::multiply(sqr_image, sqr_image, sqr_image);
        IntegralImage ii2; ii2(sqr_image);
        
        // compute the standard deviations
        int x, y, i;
        double *ddst;
        cv::Scalar z1, z2;
        for(y = 0; y < h; ++y)
        {
            r.y = y;
            ddst = out_image.ptr<double>(y);
            for(x = 0; x < w; ++x, ddst += nc)
            {
                r.x = x;
                z1 = ii.sum_rect<double>(r);
                z2 = ii2.sum_rect<double>(r);
                for(i = 0; i < nc; ++i) ddst[i] = sqrt(s*(z2[i]-s*z1[i]*z1[i]));
            }
        }
    }
}
