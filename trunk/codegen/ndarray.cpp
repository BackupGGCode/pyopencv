// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/handle.hpp>
#include <boost/python/cast.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/detail/raw_pyobject.hpp>
#include <boost/python/extract.hpp>
#include "ndarray.hpp"

namespace boost { namespace python { namespace numeric {

namespace
{
  enum state_t { failed = -1, unknown, succeeded };
  state_t state = unknown;
  std::string module_name;
  std::string type_name;

  handle<> array_module;
  handle<> array_type;
  handle<> array_function;

  void throw_load_failure()
  {
      PyErr_Format(
          PyExc_ImportError
          , "No module named '%s' or its type '%s' did not follow the NumPy protocol (1.3.0 or later)"
          , module_name.c_str(), type_name.c_str());
      throw_error_already_set();
      
  }

  bool load(bool throw_on_error)
  {
      if (!state)
      {
          if (module_name.size() == 0)
          {
              module_name = "numpy";
              type_name = "ndarray";
              if (load(false))
                  return true;
              module_name = "numarray";
              type_name = "NDArray";
          }

          state = failed;
          PyObject* module = ::PyImport_Import(object(module_name).ptr());
          if (module)
          {
              PyObject* type = ::PyObject_GetAttrString(module, const_cast<char*>(type_name.c_str()));

              if (type && PyType_Check(type))
              {
                  array_type = handle<>(type);
                  PyObject* function = ::PyObject_GetAttrString(module, const_cast<char*>("array"));
                  
                  if (function && PyCallable_Check(function))
                  {
                      array_function = handle<>(function);
                      state = succeeded;
                  }
              }
          }
      }
      
      if (state == succeeded)
          return true;
      
      if (throw_on_error)
          throw_load_failure();
      
      PyErr_Clear();
      return false;
  }

  object demand_array_function()
  {
      load(true);
      return object(array_function);
  }
}

void ndarray::set_module_and_type(char const* package_name, char const* type_attribute_name)
{
    state = unknown;
    module_name = package_name ? package_name : "" ;
    type_name = type_attribute_name ? type_attribute_name : "" ;
}

std::string ndarray::get_module_name()
{
    load(false);
    return module_name;
}

namespace aux
{
# define BOOST_PYTHON_AS_OBJECT(z, n, _) object(x##n)
# define BOOST_PP_LOCAL_MACRO(n)                                        \
    ndarray_base::ndarray_base(BOOST_PP_ENUM_PARAMS(n, object const& x))    \
        : object(demand_array_function()(BOOST_PP_ENUM_PARAMS(n, x)))   \
    {}
# define BOOST_PP_LOCAL_LIMITS (1, 6)
# include BOOST_PP_LOCAL_ITERATE()
# undef BOOST_PYTHON_AS_OBJECT

    ndarray_base::ndarray_base(BOOST_PP_ENUM_PARAMS(7, object const& x))
        : object(demand_array_function()(BOOST_PP_ENUM_PARAMS(7, x)))
    {}

  object ndarray_base::argmax(long axis)
  {
      return attr("argmax")(axis);
  }
  
  object ndarray_base::argmin(long axis)
  {
      return attr("argmin")(axis);
  }
  
  object ndarray_base::argsort(long axis)
  {
      return attr("argsort")(axis);
  }
  
  object ndarray_base::astype(object const& type)
  {
      return attr("astype")(type);
  }
  
  void ndarray_base::byteswap()
  {
      attr("byteswap")();
  }
  
  object ndarray_base::copy() const
  {
      return attr("copy")();
  }
  
  object ndarray_base::diagonal(long offset, long axis1, long axis2) const
  {
      return attr("diagonal")(offset, axis1, axis2);
  }
  
  void ndarray_base::info() const
  {
      attr("info")();
  }
  
  bool ndarray_base::is_c_array() const
  {
      return extract<bool>(attr("is_c_array")());
  }
  
  bool ndarray_base::isbyteswapped() const
  {
      return extract<bool>(attr("isbyteswapped")());
  }
  
  ndarray ndarray_base::new_(object type) const
  {
      return extract<ndarray>(attr("new")(type))();
  }
  
  void ndarray_base::sort()
  {
      attr("sort")();
  }
  
  object ndarray_base::trace(long offset, long axis1, long axis2) const
  {
      return attr("trace")(offset, axis1, axis2);
  }
  
  object ndarray_base::type() const
  {
      return attr("type")();
  }
  
  char ndarray_base::typecode() const
  {
      return extract<char>(attr("typecode")());
  }

  object ndarray_base::factory(
          object const& sequence
        , object const& typecode
        , bool copy
        , bool savespace
        , object type
        , object shape
  )
  {
      return attr("factory")(sequence, typecode, copy, savespace, type, shape);
  }

  object ndarray_base::getflat() const
  {
      return attr("getflat")();
  }
      
  long ndarray_base::getrank() const
  {
      return extract<long>(attr("getrank")());
  }
  
  object ndarray_base::getshape() const
  {
      return attr("getshape")();
  }
  
  bool ndarray_base::isaligned() const
  {
      return extract<bool>(attr("isaligned")());
  }
  
  bool ndarray_base::iscontiguous() const
  {      
      return extract<bool>(attr("iscontiguous")());
  }
  
  long ndarray_base::itemsize() const
  {
      return extract<long>(attr("itemsize")());
  }
  
  long ndarray_base::nelements() const
  {
      return extract<long>(attr("nelements")());
  }
  
  object ndarray_base::nonzero() const
  {
      return attr("nonzero")();
  }
   
  void ndarray_base::put(object const& indices, object const& values)
  {
      attr("put")(indices, values);
  }
   
  void ndarray_base::ravel()
  {
      attr("ravel")();
  }
   
  object ndarray_base::repeat(object const& repeats, long axis)
  {
      return attr("repeat")(repeats, axis);
  }
   
  void ndarray_base::resize(object const& shape)
  {
      attr("resize")(shape);
  }
      
  void ndarray_base::setflat(object const& flat)
  {
      attr("setflat")(flat);
  }
  
  void ndarray_base::setshape(object const& shape)
  {
      attr("setshape")(shape);
  }
   
  void ndarray_base::swapaxes(long axis1, long axis2)
  {
      attr("swapaxes")(axis1, axis2);
  }
   
  object ndarray_base::take(object const& sequence, long axis) const
  {
      return attr("take")(sequence, axis);
  }
   
  void ndarray_base::tofile(object const& file) const
  {
      attr("tofile")(file);
  }
   
  str ndarray_base::tostring() const
  {
      return str(attr("tostring")());
  }
   
  void ndarray_base::transpose(object const& axes)
  {
      attr("transpose")(axes);
  }
   
  object ndarray_base::view() const
  {
      return attr("view")();
  }
}

}}} // namespace boost::python::numeric
