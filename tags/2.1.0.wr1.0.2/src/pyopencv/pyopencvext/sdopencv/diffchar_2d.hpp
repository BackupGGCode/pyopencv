#ifndef SDCPP_DIFFERENTIALCHARACTERISTICS_H
#define SDCPP_DIFFERENTIALCHARACTERISTICS_H

#include <cxcore.h>

namespace sdopencv
{
    // See cv::Sobel for more details about the parameters
    class DifferentialImage
    {
        private:
            static cv::Mat Tx, Ty, Txx, Txy, Tyy;
            static void init_templates();
            
        protected:
            int max_order;
            
            int ksize;
            double scale, delta;
            int borderType;
            
            double Zx, Zy, Zxx, Zxy, Zyy; // normalization factors

            void compute_normalization_factors();
            
            cv::Size image_size;
        public: // public data
            cv::Mat Ix, Iy; // 1st order differential images
            cv::Mat Ixx, Ixy, Iyy; // 2nd order differential images

        public: // public methods

            // max_order = 1 or 2 only
            // max_order = 1: only dI/dx and dI/dy are computed
            // max_order = 2: dI/dx, dI/dy d^2I/dx^2, d^2I/dy^2, and d^2I/dxdy are computed
            DifferentialImage(int max_order = 2, int ksize=3, double scale=1, double delta=0, int borderType=4);

            void operator()(const cv::Mat &image);

            // compute the gradient magnitude squared per pixel
            // Requirement:
            //   max_order >= 1
            void gradient_magnitude_squared(cv::Mat &output);

            // compute the gradient magnitude per pixel
            // Requirement:
            //   max_order >= 1
            void gradient_magnitude(cv::Mat &output);

            // compute the gradient orientation per pixel
            // Requirement:
            //   max_order >= 1
            void gradient_orientation(cv::Mat &output);

            // compute the gradient vector per pixel
            // Requirement:
            //   max_order >= 1
            void gradient(cv::Mat &output);

            // compute the gradient vector in polar coordinates per pixel
            // Requirement:
            //   max_order >= 1
            void gradient_polar(cv::Mat &output);

            // compute the Laplacian per pixel
            // Requirement:
            //   max_order >= 2
            void laplacian(cv::Mat &output);

            // compute the Hessian matrix per pixel
            // Requirement:
            //   max_order >= 2
            void hessian(cv::Mat &output);

            // compute the curvature per pixel
            // Requirement:
            //   max_order >= 2
            void curvature(cv::Mat &output);
    };
}

#endif // SDCPP_DIFFERENTIALCHARACTERISTICS_H
