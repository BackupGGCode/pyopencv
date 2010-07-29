#ifndef SD_HIGHGUI_WRAPPER_H
#define SD_HIGHGUI_WRAPPER_H

#include "opencv_headers.hpp"

#define __OPENCV_CV_HPP__ // to turn off cv.hpp and turn on sdcv.hpp -- there's a bug
#include "cv.h"
#undef __OPENCV_CV_HPP__
#include "sdcv.hpp"

#include "highgui.h"


#include "highgui_template_instantiations.hpp"


#endif
