// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef SD_RETURN_AS_NDARRAY_HPP
# define SD_RETURN_AS_NDARRAY_HPP

# include <boost/python/detail/prefix.hpp>

# include <boost/python/default_call_policies.hpp>
# include <boost/python/object.hpp>
# include <boost/python/extract.hpp>
# include <algorithm>

# include "opencv_extra.hpp"

# include <iostream>

namespace boost { namespace python { 

template <> struct to_python_value<const cv::Mat &>
{
    inline PyObject* operator()(cv::Mat const& x) const
    {
        std::cout << "Everthing is fine until here." << std::endl;
        object obj;
        convert_ndarray_from(x, obj);
        std::cout << "Everthing is fine until here." << std::endl;
        return bp::incref(obj.ptr());
    }
    inline PyTypeObject const* get_pytype() const
    {
        return 0;//PyArray_Type;
    }
};


}} // namespace boost::python

#endif // SD_RETURN_AS_NDARRAY_HPP
