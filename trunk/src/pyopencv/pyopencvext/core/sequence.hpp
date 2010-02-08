// Copyright Minh-Tri Pham 2010.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef SD_SEQUENCE_HPP
# define SD_SEQUENCE_HPP

# include <boost/python/detail/prefix.hpp>

# include <boost/python/tuple.hpp>
# include <boost/python/str.hpp>
# include <boost/preprocessor/iteration/local.hpp>
# include <boost/preprocessor/cat.hpp>
# include <boost/preprocessor/repetition/enum.hpp>
# include <boost/preprocessor/repetition/enum_params.hpp>
# include <boost/preprocessor/repetition/enum_binary_params.hpp>

namespace boost { namespace python {

namespace aux
{
    struct sequence_object_manager_traits
    {
        static bool check(PyObject* obj);
        static detail::new_non_null_reference adopt(PyObject* obj);
        static PyTypeObject const* get_pytype() ;
    };
} // namespace aux

struct sequence : object
{
public:
    sequence() : object() {}
    template <class T>
    explicit sequence(T const& x) : object(x) {}

    void check() const;

    int len() const;
    
public: // implementation detail - do not touch.
    BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(sequence, object);
};


namespace converter
{
  template <>
  struct object_manager_traits< sequence >
      : aux::sequence_object_manager_traits
  {
      BOOST_STATIC_CONSTANT(bool, is_specialized = true);
  };
}


}} // namespace boost::python

#endif // SD_SEQUENCE_HPP
