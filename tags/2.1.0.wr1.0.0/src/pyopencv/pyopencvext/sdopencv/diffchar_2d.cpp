#include <cmath>
#include <cv.h>

#include "dtype.hpp"
#include "diffchar_2d.hpp"

using namespace std;

namespace sdopencv
{
    cv::Mat DifferentialImage::Tx, DifferentialImage::Ty, DifferentialImage::Txx, DifferentialImage::Txy, DifferentialImage::Tyy;

    void DifferentialImage::init_templates()
    {
        static bool templates_inited = false;
        if(templates_inited) return;
        
        int y, x;
        double *d;
        
        // Tx
        Tx.create(64, 64, CV_64FC1);
        for(y =0; y < 64; ++y) { d = Tx.ptr<double>(y); for(x = 0; x < 64; ++x) d[x] = x; }
        
        // Ty
        Ty.create(64, 64, CV_64FC1);
        for(y =0; y < 64; ++y) { d = Ty.ptr<double>(y); for(x = 0; x < 64; ++x) d[x] = y; }

        // Txx
        Txx.create(64, 64, CV_64FC1);
        for(y =0; y < 64; ++y) { d = Txx.ptr<double>(y); for(x = 0; x < 64; ++x) d[x] = 0.5*x*x; }

        // Txy
        Txy.create(64, 64, CV_64FC1);
        for(y =0; y < 64; ++y) { d = Txy.ptr<double>(y); for(x = 0; x < 64; ++x) d[x] = x*y; }

        // Tyy
        Tyy.create(64, 64, CV_64FC1);
        for(y =0; y < 64; ++y) { d = Tyy.ptr<double>(y); for(x = 0; x < 64; ++x) d[x] = 0.5*y*y; }

        templates_inited = true;
    }

    
    void DifferentialImage::compute_normalization_factors()
    {
        cv::Mat tmp(64, 64, CV_64FC1);        
        if(max_order >= 1)
        {
            cv::Sobel(Tx, tmp, CV_64F, 1, 0, ksize, scale, delta, borderType);            
            Zx = 1.0/tmp.at<double>(32, 32);
            cv::Sobel(Ty, tmp, CV_64F, 0, 1, ksize, scale, delta, borderType);            
            Zy = 1.0/tmp.at<double>(32, 32);
        }
        if(max_order >= 2)
        {
            cv::Sobel(Txx, tmp, CV_64F, 2, 0, ksize, scale, delta, borderType);            
            Zxx = 1.0/tmp.at<double>(32, 32);
            cv::Sobel(Txy, tmp, CV_64F, 1, 1, ksize, scale, delta, borderType);            
            Zxy = 1.0/tmp.at<double>(32, 32);
            cv::Sobel(Tyy, tmp, CV_64F, 0, 2, ksize, scale, delta, borderType);            
            Zyy = 1.0/tmp.at<double>(32, 32);
        }
    }

    
    // max_order = 1 or 2 only
    // max_order = 1: additionally dI/dx and dI/dy are computed
    // max_order = 2: additionally d^2I/dx^2, d^2I/dy^2, and d^2I/dxdy are computed    
    DifferentialImage::DifferentialImage(int _max_order, int _ksize, double _scale, double _delta, int _borderType)
        : max_order(_max_order), ksize(_ksize), scale(_scale), delta(_delta), borderType(_borderType)
    {
        if(max_order < 1 || max_order > 2) CV_Error( -1, "max_order must be either 1 or 2" );        
        init_templates();
        compute_normalization_factors();
    }


    void DifferentialImage::operator()(const cv::Mat &image)
    {
        if(image.channels() > 1) CV_Error( -1, "Unsupported multi-channel images" );
        
        image_size = image.size();

        if(max_order >= 1)
        {
            // dI/dx
            Ix.create(image_size, CV_64FC1);
            cv::Sobel(image, Ix, CV_64F, 1, 0, ksize, scale, delta, borderType);
            Ix.convertTo(Ix, CV_64F, Zx);
            
            // dI/dy
            Iy.create(image_size, CV_64FC1);
            cv::Sobel(image, Iy, CV_64F, 0, 1, ksize, scale, delta, borderType);
            Iy.convertTo(Iy, CV_64F, Zy);
        }

        if(max_order >= 2)
        {
            // d^2I/dx^2
            Ixx.create(image_size, CV_64FC1);
            cv::Sobel(image, Ixx, CV_64F, 2, 0, ksize, scale, delta, borderType);
            Ixx.convertTo(Ixx, CV_64F, Zxx);
            
            // d^2I/dxdy
            Ixy.create(image_size, CV_64FC1);
            cv::Sobel(image, Ixy, CV_64F, 1, 1, ksize, scale, delta, borderType);
            Ixy.convertTo(Ixy, CV_64F, Zxy);
            
            // d^2I/dy^2
            Iyy.create(image_size, CV_64FC1);
            cv::Sobel(image, Iyy, CV_64F, 0, 2, ksize, scale, delta, borderType);
            Iyy.convertTo(Iyy, CV_64F, Zyy);
        }
    }

    
    // compute the gradient magnitude squared per pixel
    // Requirement:
    //   max_order >= 1
    void DifferentialImage::gradient_magnitude_squared(cv::Mat &output)
    {
        output.create(image_size, CV_64FC1);

        double *dIx, *dIy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIx = (double *)Ix.ptr(y);
            dIy = (double *)Iy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
                ddst[x] = dIx[x]*dIx[x] + dIy[x]*dIy[x];
        }
    }


    // compute the gradient magnitude per pixel
    // Requirement:
    //   max_order >= 1
    void DifferentialImage::gradient_magnitude(cv::Mat &output)
    {
        output.create(image_size, CV_64FC1);

        double *dIx, *dIy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIx = (double *)Ix.ptr(y);
            dIy = (double *)Iy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
                ddst[x] = hypot(dIx[x], dIy[x]);
        }
    }


    // compute the gradient orientation per pixel
    // Requirement:
    //   max_order >= 1
    void DifferentialImage::gradient_orientation(cv::Mat &output)
    {
        output.create(image_size, CV_64FC1);

        double *dIx, *dIy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIx = (double *)Ix.ptr(y);
            dIy = (double *)Iy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
                ddst[x] = atan2(dIy[x], dIx[x]);
        }
    }
    

    // compute the gradient vector per pixel
    // Requirement:
    //   max_order >= 1
    void DifferentialImage::gradient(cv::Mat &output)
    {
        cv::Mat arr[2];
        arr[0] = Ix; arr[1] = Iy;
        output.create(image_size, CV_64FC2);
        cv::merge(arr, 2, output);
    }
    

    // compute the gradient vector in polar coordinates per pixel
    // Requirement:
    //   max_order >= 1
    void DifferentialImage::gradient_polar(cv::Mat &output)
    {
        output.create(image_size, CV_64FC2);

        double *dIx, *dIy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIx = (double *)Ix.ptr(y);
            dIy = (double *)Iy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
            {
                ddst[x*2] = hypot(dIx[x], dIy[x]);
                ddst[x*2+1] = atan2(dIy[x], dIx[x]);
            }
        }
    }
    

    // compute the Laplacian per pixel
    // Requirement:
    //   max_order >= 2
    void DifferentialImage::laplacian(cv::Mat &output)
    {
        if(max_order < 2) CV_Error( -1, "max_order must be >= 2" );
        output.create(image_size, CV_64FC1);

        double *dIxx, *dIyy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIxx = (double *)Ixx.ptr(y);
            dIyy = (double *)Iyy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
                ddst[x] = dIxx[x] + dIyy[x];
        }
    }

    
    // compute the Hessian matrix per pixel
    // Requirement:
    //   max_order >= 2
    void DifferentialImage::hessian(cv::Mat &output)
    {
        cv::Mat arr[4];
        arr[0] = Ixx; arr[1] = Ixy; arr[2] = Ixy; arr[3] = Iyy;
        output.create(image_size, CV_64FC4);
        cv::merge(arr, 4, output);
    }


    // compute the curvature per pixel
    // Requirement:
    //   max_order >= 2
    void DifferentialImage::curvature(cv::Mat &output)
    {
        if(max_order < 2) CV_Error( -1, "max_order must be >= 2" );
        output.create(image_size, CV_64FC1);

        double *dIx, *dIy, *dIxx, *dIxy, *dIyy;
        double *ddst;
        for(int y = 0; y < image_size.height; ++y)
        {
            dIx = (double *)Ix.ptr(y);
            dIy = (double *)Iy.ptr(y);
            dIxx = (double *)Ixx.ptr(y);
            dIxy = (double *)Ixy.ptr(y);
            dIyy = (double *)Iyy.ptr(y);
            ddst = (double *)output.ptr(y);

            int x = image_size.width; while(--x >= 0)
            {
                double fx = dIx[x], fy = dIy[x];
                double g = fx*fx+fy*fy;
                if(abs(g) > 1E-10)
                    ddst[x] = (2*fx*fy*dIxy[x] - fx*fx*dIyy[x] - fy*fy*dIxx[x]) * pow(g, -1.5);
                else
                    ddst[x] = 0;
            }
        }
    }

}
