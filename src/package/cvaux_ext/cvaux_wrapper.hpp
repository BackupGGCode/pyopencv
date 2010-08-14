#ifndef SD_CVAUX_WRAPPER_HPP
#define SD_CVAUX_WRAPPER_HPP

#include "opencv_headers.hpp"

struct CvFileStorage {};

#define __OPENCV_CV_HPP__ // to turn off cv.hpp and turn on sdcv.hpp -- there's a bug
#include "cv.h"
#undef __OPENCV_CV_HPP__
#include "sdcv.hpp"

#include "cvaux.h"

#include "cvaux_template_instantiations.hpp"


#endif
