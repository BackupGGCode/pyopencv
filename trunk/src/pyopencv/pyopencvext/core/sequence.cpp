// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/handle.hpp>
#include <boost/python/cast.hpp>
#include <boost/python/ptr.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/detail/raw_pyobject.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/object/life_support.hpp>

#include "sequence.hpp"
#include "ndarray.hpp"

namespace bp = boost::python;

namespace sdcpp {

void sequence::check_obj(object const &obj) const
{
    if(!check<sequence>(obj))
    {
        PyErr_SetString(PyExc_TypeError, "Not a sequence.");
        throw bp::error_already_set();
    }
}

int sequence::len() const { return bp::len(get_obj()); }

template<> bool check<sequence>(object const &obj)
{
    if(check<ndarray>(obj)) return true;
    PyObject *ptr = obj.ptr();
    return ptr == Py_None || PyTuple_Check(ptr) || PyList_Check(ptr) || PyString_Check(ptr);
}

} // namespace sdcpp
