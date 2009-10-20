#include "opencv.hpp"

#include "boost/python.hpp"
#include "boost/python/object.hpp"

#include <iostream>

namespace bp = boost::python;

void CV_CDECL sdTrackbarCallback2(int pos, void* userdata)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)userdata)));
    (items[0])(pos, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}


void CV_CDECL sdMouseCallback(int event, int x, int y, int flags, void* param)
{
    bp::object items(bp::handle<>(bp::borrowed((PyObject *)param)));
    (items[0])(event, x, y, flags, bp::object(items[1])); // need a copy of items[1] to make it safe with threading
}


