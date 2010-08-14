// Copyright Minh-Tri Pham 2009.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef SD_WITH_OWNERSHIPLEVEL_POSTCALL_HPP
# define SD_WITH_OWNERSHIPLEVEL_POSTCALL_HPP

# include <boost/python/detail/prefix.hpp>

# include <boost/python/default_call_policies.hpp>
# include <boost/python/object.hpp>
# include <algorithm>

namespace boost { namespace python { 

template <
    std::size_t ownershiplevel = 0
  , class BasePolicy_ = default_call_policies
>
struct with_ownershiplevel_postcall : BasePolicy_
{
    BOOST_STATIC_ASSERT(ownershiplevel >= 0);

    template <class ArgumentPackage>
    static PyObject* postcall(ArgumentPackage const& args_, PyObject* result)
    {
        if(result != Py_None) object(handle<>(borrowed(result))).attr("_ownershiplevel") = ownershiplevel;
        return result;
    }
};


}} // namespace boost::python

#endif // SD_WITH_OWNERSHIPLEVEL_POSTCALL_HPP
