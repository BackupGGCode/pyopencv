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
#include "ndarray.hpp"

#include "sequence.hpp"

namespace bp = boost::python;

namespace boost { namespace python {

// ================================================================================================

namespace aux
{
    bool sequence_object_manager_traits::check(PyObject* obj)
    {
        return array_object_manager_traits::check(obj) || obj == Py_None ||
            PyTuple_Check(obj) || PyList_Check(obj) || PyString_Check(obj);
    }

    python::detail::new_non_null_reference
    sequence_object_manager_traits::adopt(PyObject* obj)
    {
        return detail::new_non_null_reference(
        true); // let's return true for now
    }

    PyTypeObject const* sequence_object_manager_traits::get_pytype()
    {
        return &PyTuple_Type; // let's use PyTuple_Type for now
    }
}

void sequence::check() const
{
    if(!aux::sequence_object_manager_traits::check(ptr()))
    {
        PyErr_SetString(PyExc_TypeError, "The variable is not a sequence.");
        throw error_already_set(); 
    }
}

int sequence::len() const { check(); return bp::len(*this); }


// ================================================================================================

}} // namespace boost::python
