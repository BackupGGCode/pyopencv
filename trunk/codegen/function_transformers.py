from pygccxml import declarations
from pyplusplus import code_repository
from pyplusplus.function_transformers import *
import pyplusplus.function_transformers.transformers as _T


def expose_addressof_member(klass, member_name, exclude_member=True):
    klass.include_files.append( "boost/python/long.hpp" )
    if exclude_member:
        klass.var(member_name).exclude()
    klass.add_declaration_code('''
    boost::python::long_ get_MEMBER_NAME_addr( CLASS_TYPE const & inst ){
        return boost::python::long_((int)&inst.MEMBER_NAME);
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('def( "get_MEMBER_NAME_addr", &::get_MEMBER_NAME_addr )'.replace("MEMBER_NAME", member_name))
    # TODO: finish the wrapping with ctypes code
    
def expose_member_as_str(klass, member_name):
    klass.include_files.append( "boost/python/object.hpp" )
    klass.include_files.append( "boost/python/str.hpp" )
    klass.var(member_name).exclude()
    klass.add_wrapper_code('''
    static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){        
        return inst.MEMBER_NAME? bp::str(inst.MEMBER_NAME): bp::object();
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    
def expose_member_as_pointee(klass, member_name):
    klass.include_files.append( "boost/python/object.hpp" )
    klass.var(member_name).exclude()
    klass.add_wrapper_code('''
    static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){        
        return inst.MEMBER_NAME? bp::object(inst.MEMBER_NAME): bp::object();
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    

def remove_ptr( type_ ):
    if declarations.is_pointer( type_ ):
        return declarations.remove_pointer( type_ )
    else:
        raise TypeError( 'Type should be a pointer, got %s.' % type_ )


# -----------------------------------------------------------------------------------------------
# Function transfomers
# -----------------------------------------------------------------------------------------------

# input_double_pointee_t
class input_double_pointee_t(transformer_t):
    """Handles a double pointee input.
    
    Convert by dereferencing: do_smth(your_type **v) -> do_smth(your_type v)

    Right now compiler should be able to use implicit conversion
    """

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        if not declarations.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_double_pointee_t" transformation, argument %s type must be a pointer or a array (got %s).' ) \
                  % ( function, self.arg_ref.name, arg.type)

    def __str__(self):
        return "input_double_pointee(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        tmp_type = remove_ptr( self.arg.type )
        w_arg.type = remove_ptr( tmp_type )
        if not declarations.is_convertible( w_arg.type, self.arg.type ):
            controller.add_pre_call_code("%s tmp_%s = reinterpret_cast< %s >(& %s);" % ( tmp_type, w_arg.name, tmp_type, w_arg.name ))
            casting_code = 'reinterpret_cast< %s >( & tmp_%s )' % (self.arg.type, w_arg.name)
            controller.modify_arg_expression(self.arg_index, casting_code)

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return []

def input_double_pointee( *args, **keywd ):
    def creator( function ):
        return input_double_pointee_t( function, *args, **keywd )
    return creator

# input_smart_pointee_t
class input_smart_pointee_t(transformer_t):
    """Handles a pointee input.
    
    Convert by dereferencing: do_smth(your_type *v) -> do_smth(object v2)
    where v2 is either of type NoneType or type 'your_type'. 
    If v2 is None, v is NULL.  Otherwise, v is the pointer to v2.
    """

    def __init__(self, function, arg_ref):
        transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_smart_pointee(%s)" % self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ "boost/python/object.hpp", "boost/python/extract.hpp" ]

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        data_type = declarations.remove_const(remove_ptr( self.arg.type ))
        w_arg.type = declarations.dummy_type_t( "boost::python::object" )
        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = 'bp::object()'
        controller.add_pre_call_code("%s const &tmp_%s = bp::extract<%s const &>(%s);" % (data_type, w_arg.name, data_type, w_arg.name))
        controller.modify_arg_expression(self.arg_index, "(%s.ptr() != Py_None)? (%s)(&tmp_%s): 0" % (w_arg.name, self.arg.type, w_arg.name))

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_smart_pointee( *args, **keywd ):
    def creator( function ):
        return input_smart_pointee_t( function, *args, **keywd )
    return creator
    
# input_string_t
class input_string_t(transformer_t):
    """Handles a string.
    
    Convert: do_smth(void *v) -> do_smth(str v2)
    where v2 is a Python string and v is a pointer to its content.
    If vs is None, then v is NULL.
    """

    def __init__(self, function, arg_ref):
        transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        if not declarations.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_string_t" transformation, argument %s type must be a pointer (got %s).' ) \
                  % ( function, arg_ref, self.arg.type)

    def __str__(self):
        return "input_string(%s)" % self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ "boost/python/str.hpp", "boost/python/object.hpp", "boost/python/extract.hpp" ]

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        w_arg.type = declarations.dummy_type_t( "boost::python::object" )
        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = 'bp::object()'
        controller.modify_arg_expression(self.arg_index, "(%s.ptr() != Py_None)? (void *)((const char *)bp::extract<const char *>(%s)): 0" % (w_arg.name, w_arg.name))

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_string( *args, **keywd ):
    def creator( function ):
        return input_string_t( function, *args, **keywd )
    return creator

    
class input_dynamic_array_t(transformer.transformer_t):
    """Handles an input array with a dynamic size.

    void do_something([int N, ]double* v) ->  do_something(object v2)

    where v2 is a Python sequence of N items. Each item is of the same type as v's element type.
    """

    def __init__(self, function, arg_ref, arg_size_ref=None):
        """Constructor.

        """
        transformer.transformer_t.__init__( self, function )

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _T.is_ptr_or_array( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_dynamic_array" transformation, argument %s type must be a array or a pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)

        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
            
            if not declarations.is_integral( self.arg_size.type ):
                raise ValueError( '%s\nin order to use "input_dynamic_array" transformation, argument %s type must be an integer (got %s).' ) \
                      % ( function, self.arg_size.name, self.arg_size.type)

        else:
            self.arg_size = None

        self.array_item_type = declarations.remove_const( declarations.array_item_type( self.arg.type ) )

    def __str__(self):
        if self.arg_size is not None:
            return "input_dynamic_array(%s,%d)"%( self.arg.name, self.arg_size.name)
        return "input_dynamic_array(%s)"% self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ code_repository.convenience.file_name ]

    def __configure_sealed(self, controller):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        w_arg.type = declarations.dummy_type_t( "boost::python::object" )

        if self.arg_size is not None:
            #removing arg_size from the function wrapper definition
            controller.remove_wrapper_arg( self.arg_size.name )
        
        # array size
        array_size = controller.declare_variable( "int", "size_" + self.arg.name, 
            "=bp::len(%s)" % self.arg.name)

        # Declare a variable that will hold the C array...
        native_array = controller.declare_variable( declarations.pointer_t(self.array_item_type), 
            "native_" + self.arg.name, 
            "= new %s [%s]" % (self.array_item_type, array_size) )
            
        controller.add_post_call_code("delete[] %s;" % native_array)

        copy_pylist2arr = _T._seq2arr.substitute( type=self.array_item_type
                                                , pylist=w_arg.name
                                                , array_size=array_size
                                                , native_array=native_array )

        controller.add_pre_call_code( copy_pylist2arr )

        controller.modify_arg_expression( self.arg_index, native_array )
        if self.arg_size is not None:
            controller.modify_arg_expression( self.arg_size_index, array_size )

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def __configure_v_mem_fun_override( self, controller ):
        global _arr2seq
        pylist = controller.declare_py_variable( declarations.dummy_type_t( 'boost::python::list' )
                                                 , 'py_' + self.arg.name )

        copy_arr2pylist = _arr2seq.substitute( native_array=self.arg.name
                                                , array_size=self.array_size
                                                , pylist=pylist )

        controller.add_py_pre_call_code( copy_arr2pylist )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_override( controller.override_controller )
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_dynamic_array( *args, **keywd ):
    def creator( function ):
        return input_dynamic_array_t( function, *args, **keywd )
    return creator


