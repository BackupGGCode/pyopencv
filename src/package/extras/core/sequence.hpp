// Copyright Minh-Tri Pham 2010.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef SD_SEQUENCE_HPP
# define SD_SEQUENCE_HPP

# include <boost/python.hpp>
# include <boost/python/object.hpp>
# include <boost/python/detail/prefix.hpp>

# include <boost/python/tuple.hpp>
# include <boost/python/str.hpp>
# include <boost/preprocessor/iteration/local.hpp>
# include <boost/preprocessor/cat.hpp>
# include <boost/preprocessor/repetition/enum.hpp>
# include <boost/preprocessor/repetition/enum_params.hpp>
# include <boost/preprocessor/repetition/enum_binary_params.hpp>

# include <vector>

namespace sdcpp {

using namespace boost::python;

// ----------------------------------------------------------------------------------------------
// get an object which is a _borrowed_ reference to py_obj
// borrowed vs new: http://docs.python.org/release/2.5.2/ext/refcountsInPython.html
inline object get_borrowed_object(PyObject *py_obj) { return object(handle<>(borrowed(py_obj))); }

// get an object which is a _new_ reference to py_obj
// borrowed vs new: http://docs.python.org/release/2.5.2/ext/refcountsInPython.html
inline object get_new_object(PyObject *py_obj) { return object(handle<>(py_obj)); }

// ----------------------------------------------------------------------------------------------
// structure describing how data are arranged in a dense array
// dimension 0 corresponds to the highest dimension
// dimension ndim-1 corresponds to the lowest dimension
// This big-endian arrangement is chosen so that it matches with that of numpy
struct array_data_arrangement
{
    int ndim; // number of dimensions
    Py_intptr_t item_size; // size of one item/element, in byte
    Py_intptr_t total_size; // size of the data, in byte
    std::vector<Py_intptr_t> size; // number of elements per dimension
    std::vector<Py_intptr_t> stride; // element size per dimension, in byte
};


// ----------------------------------------------------------------------------------------------
template<typename T> inline bool check(object const &obj) { return true; }
template<typename T> inline PyTypeObject const *get_pytype() { return 0; }

class sdobject
{
public:
    sdobject(object const &obj) : obj(obj) { incref(obj.ptr()); }
    ~sdobject() { decref(obj.ptr()); }
    
    object const & get_obj() const { return obj; }
    sdobject &operator=(sdobject const &inst)
    {
        incref(inst.obj.ptr());
        decref(obj.ptr());
        obj = inst.obj;
        return *this;
    }

protected:
    object obj;
};

template<typename SDOBJECT>
struct to_python
{
    static PyObject *convert(SDOBJECT const &inst) { return incref(inst.get_obj().ptr()); }
};

template<typename SDOBJECT>
void register_sdobject()
{
    struct from_python
    {
        static void *convertible(PyObject *py_obj)
        {
            return check<SDOBJECT>(get_borrowed_object(py_obj))? py_obj: 0;
        }
        
        static void construct(PyObject *py_obj, converter::rvalue_from_python_stage1_data *data)
        {
            typedef converter::rvalue_from_python_storage<SDOBJECT> storage_t;
            storage_t* the_storage = reinterpret_cast<storage_t*>( data );
            void* memory_chunk = the_storage->storage.bytes;
            new (memory_chunk) SDOBJECT(get_borrowed_object(py_obj));
            data->convertible = memory_chunk;
        }
        
        static PyTypeObject const *expected_pytype()
        {
            return get_pytype<SDOBJECT>();
        }
    };
    
    to_python_converter< SDOBJECT, to_python<SDOBJECT> >();

    converter::registry::push_back( &from_python::convertible, &from_python::construct,
        type_id<SDOBJECT>(), &from_python::expected_pytype );
}

class sequence : public sdobject
{
public:
    sequence(object const &obj) : sdobject(obj) { check_obj(obj); }
    sequence &operator=(sequence const &inst)
    {
        sdobject::operator=(inst);
        return *this;
    }
    
    void check_obj(object const &obj) const;
    int len() const;
};

template<> bool check<sequence>(object const &obj);
template<> inline PyTypeObject const *get_pytype<sequence>() { return &PyList_Type; }

} // namespace sdcpp


#endif // SD_SEQUENCE_HPP
