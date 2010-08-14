#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------


from pygccxml import declarations as _D
from pyplusplus import code_repository
from pyplusplus.function_transformers import *
import pyplusplus.function_transformers.transformers as _T
from pyplusplus.decl_wrappers import call_policies as CP

import common
from memvar_transformers import *


def expose_func(func, ownershiplevel=None, ward_indices=(), return_arg_index=None, return_pointee=True, transformer_creators=[]):
    
    func.include()    
    func._transformer_creators.extend(transformer_creators)
    func.set_exportable(True) # make sure the function is exposed even if there might be a compilation error
    
    cp = CP.return_value_policy(CP.reference_existing_object) if return_pointee is True else None
    
    if return_arg_index is None:
        for ward_index in ward_indices:
            cp = CP.with_custodian_and_ward_postcall(0, ward_index, cp)
    else:
        cp = CP.return_arg(return_arg_index) # ignore previous settings
        for ward_index in ward_indices:
            cp = CP.with_custodian_and_ward(return_arg_index, ward_index, cp)
            
    if ownershiplevel is not None:
        cp = with_ownershiplevel_postcall(ownershiplevel, cp)
        
    if cp is not None:
        func.call_policies = cp


# -----------------------------------------------------------------------------------------------
# Call policies
# -----------------------------------------------------------------------------------------------

class with_ownershiplevel_postcall_t( CP.compound_policy_t ):
    """implements code generation for boost::python::with_ownershiplevel_postcall call policies"""
    def __init__( self, value=0, base=None):
        CP.compound_policy_t.__init__( self, base )
        self._value = value

    def _get_value( self ):
        return self._value
    def _set_value( self, new_value):
        self._value = new_value
    value = property( _get_value, _set_value )

    def _get_name(self, function_creator):
        return '::boost::python::with_ownershiplevel_postcall'

    def _get_args(self, function_creator):
        return [ str( self.value ) ]

    @property
    def header_file(self):
        """return a name of the header file the call policy is defined in"""
        return "with_ownershiplevel_postcall.hpp"

def with_ownershiplevel_postcall( arg_value=0, base=None):
    """create boost::python::with_ownershiplevel_postcall call policies code generator"""
    return with_ownershiplevel_postcall_t( arg_value, base )





# -----------------------------------------------------------------------------------------------
# Some functions
# -----------------------------------------------------------------------------------------------


def add_underscore(decl):
    decl.rename('_'+decl.name)
    decl.include()

def expose_lshift(klass, conversion_code, func_name="__lshift__"):
    try:
        klass.operators('<<').exclude()
    except RunError:
        pass
    klass.include_files.append("opencv_extra.hpp")
    klass.add_wrapper_code('bp::object lshift( bp::object const & other ){%s}' % conversion_code)
    klass.add_registration_code('def( "%s", &%s_wrapper::lshift )' % (func_name, klass.name))
    
def expose_rshift(klass, conversion_code, func_name="read"):
    try:
        klass.operators('>>').exclude()
    except RunError:
        pass
    klass.include_files.append("opencv_extra.hpp")
    klass.add_wrapper_code('bp::object %s_other;' % func_name)
    conversion_code = 'bp::object FUNC_NAME(){%s}' % conversion_code
    klass.add_wrapper_code(conversion_code.replace('FUNC_NAME', func_name))
    klass.add_registration_code('def( "FUNC_NAME", &KLASS_NAME_wrapper::FUNC_NAME )'\
        .replace('FUNC_NAME', func_name).replace('KLASS_NAME', klass.name))
    
def remove_ptr( type_ ):
    if _D.is_pointer( type_ ):
        return _D.remove_pointer( type_ )
    else:
        raise TypeError( 'Type should be a pointer, got %s.' % type_ )


# -----------------------------------------------------------------------------------------------
# Doc functions
# -----------------------------------------------------------------------------------------------


# some doc functions
def doc_common(func, func_arg, type_str=None):
    common.add_func_arg_doc(func, func_arg, "C++ type: %s" % func_arg.type.partial_decl_string)
    if type_str is None:
        new_type = _D.remove_const(_D.remove_reference(func_arg.type))
        type_str = common.current_sb.get_decl_equivname(new_type.partial_decl_string[2:])
    common.add_func_arg_doc(func, func_arg, "Python type: %s" % type_str)


def doc_list_of_Mat(func, func_arg):
    doc_common(func, func_arg, "list of Mat, e.g. [Mat(), Mat(), Mat()]")    
    
def doc_list_of_MatND(func, func_arg):
    doc_common(func, func_arg, "list of MatND, e.g. [MatND(), MatND(), MatND()]")    
    
def doc_Mat(func, func_arg, from_list=False):
    doc_common(func, func_arg, "Mat")
    if from_list:
        common.add_func_arg_doc(func, func_arg, "Invoke asMat() to convert a 1D Python sequence into a Mat, e.g. asMat([0,1,2]) or asMat((0,1,2)).")

def doc_list(func, func_arg):
    doc_common(func, func_arg)
    common.add_func_arg_doc(func, func_arg, "To convert a Mat into a list, invoke one of Mat's member functions to_list_of_...().")
    
def doc_output(func, func_arg):
    common.add_func_arg_doc(func, func_arg, "Output argument: omitted from input and returned as output.")
    func._output_args.append(func_arg)

def doc_dependent(func, func_arg, func_parent_arg):
    common.add_func_arg_doc(func, func_arg, "Dependent argument: omitted from input. Its value is derived from argument '%s'." % func_parent_arg.name)


# -----------------------------------------------------------------------------------------------
# Declarations
# -----------------------------------------------------------------------------------------------


def get_array_item_type(array_type):
    return _D.remove_const(_D.array_item_type(array_type))


def get_vector_pds(elem_type_pds):
    elem_type_pds = common.unique_pds(elem_type_pds)
    try:
        elem_type_pds = common.c2cpp[elem_type_pds]
    except KeyError:
        pass
    return common.unique_pds('std::vector< %s >' % elem_type_pds)

def get_vector_elem_type(vector_type):
    """Returns the element type of a std::vector type."""
    s = vector_type.decl_string
    if 'std::allocator' not in s:
        s = s[s.find('<')+1:-2]
    else:
        s = s[14:s.find(', std::allocator')] # cut all the std::allocators
        if s.startswith('std::vector'): # assume vector2d
            s = '::' + s + ', std::allocator<' + s[12:] + ' >  >'
    return _D.dummy_type_t(s)
    
def is_elem_type_fixed_size(elem_type_pds):
    """Checks if an element type is a fixed-size array-like data type."""
    if '*' in elem_type_pds:
        return False
    for t in ('char', 'unsigned char', 'short', 'unsigned short', 'int',
        'unsigned int', 'long', 'unsigned long', 'float', 'double',
        'cv::Vec', 'cv::Point', 'cv::Rect', 'cv::RotatedRect', 
        'cv::Scalar', 'cv::Range'):
        if elem_type_pds.startswith(t):
            return True
    return False
    

# -----------------------------------------------------------------------------------------------
# Function transfomers
# -----------------------------------------------------------------------------------------------



# fix_type
def fix_type(arg, type_str):
    return modify_type(arg, lambda x: _D.dummy_type_t(type_str))

# input_double_pointee_t
class input_double_pointee_t(transformer_t):
    """Handles a double pointee input.
    
    Convert by dereferencing: do_smth(your_type **v) -> do_smth(your_type &v)

    Right now compiler should be able to use implicit conversion
    """

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        if not _D.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_double_pointee_t" transformation, argument %s type must be a pointer or a array (got %s).' ) \
                  % ( function, self.arg_ref.name, arg.type)

    def __str__(self):
        return "input_double_pointee(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        tmp_type = remove_ptr( self.arg.type )
        elem_type = _D.remove_const(remove_ptr(tmp_type))
        w_arg.type = _D.reference_t(elem_type)
        z = controller.declare_variable(tmp_type, self.arg.name, "=(%s)&%s" % (tmp_type.partial_decl_string, w_arg.name))
        controller.modify_arg_expression(self.arg_index, "(%s)&%s" % (self.arg.type.partial_decl_string, z))
        # documentation
        doc_common(self.function, self.arg, common.current_sb.get_decl_equivname(elem_type.partial_decl_string))

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
        if not _D.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_string_t" transformation, argument %s type must be a pointer (got %s).' ) \
                  % ( function, arg_ref, self.arg.type)

    def __str__(self):
        return "input_string(%s)" % self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ "boost/python/str.hpp", "boost/python/object.hpp", "boost/python/extract.hpp" ]

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        w_arg.type = _D.dummy_type_t( "const char *" )
        controller.modify_arg_expression(self.arg_index, "((%s) %s)" % (self.arg.type, w_arg.name))
        # documentation
        doc_common(self.function, self.arg, "string")

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

    
# input_array1d_t
class input_array1d_t(transformer.transformer_t):
    """Handles an input array with a dynamic size.

    void do_something([int N, ]data_type* v) ->  do_something(object v2)

    where v2 is a std::vector object of N elements, whose element type is 'data_type'.
    output_arrays : set of arguments (which are arrays) to be returned as output std::vectors
        output_arrays is a list of (key,value) pairs. A key is an output argument's name. Its associated value is the number of times that the array's size is multiplied with.
    """

    def __init__(self, function, arg_ref, arg_size_ref=None, remove_arg_size=True, output_arrays={}):
        transformer.transformer_t.__init__( self, function )

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _T.is_ptr_or_array( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_array1d" transformation, argument %s type must be a array or a pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)

        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
            
            if not _D.is_integral( self.arg_size.type ):
                raise ValueError( '%s\nin order to use "input_array1d" transformation, argument %s type must be an integer (got %s).' ) \
                      % ( function, self.arg_size.name, self.arg_size.type)

        else:
            self.arg_size = None

        self.array_item_type = get_array_item_type(self.arg.type)
        self.remove_arg_size = remove_arg_size

        self.output_arrays = output_arrays

    def __str__(self):
        if self.arg_size is not None:
            return "input_array1d(%s,%s)"%( self.arg.name, self.arg_size.name)
        return "input_array1d(%s)"% self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["opencv_converters.hpp"]

    def __configure_sealed(self, controller):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        
        vec_pds = get_vector_pds(self.array_item_type.partial_decl_string)
        w_arg.type = _D.dummy_type_t( vec_pds+" const &" )
        
        # what about non-NULL default value?
        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = vec_pds+'()'
        
        # input array
        controller.modify_arg_expression( self.arg_index, "(%s)(&%s[0])" \
            % (self.arg.type.partial_decl_string, w_arg.name) )
        doc_common(self.function, self.arg, common.current_sb.get_registered_decl_name(vec_pds))

        # number of elements
        if self.remove_arg_size or self.arg_size is None:
            l_arr = "(int)(%s.size())" % w_arg.name
            if len(self.output_arrays)+(self.arg_size is not None) >= 2:
                l_arr = controller.declare_variable( _D.dummy_type_t('int'), self.arg.name, "="+l_arr )
        else:
            l_arr = self.arg_size.name
            
        # remove arg_size from the function wrapper definition and automatically fill in the missing argument
        if self.remove_arg_size and self.arg_size is not None:
            controller.remove_wrapper_arg( self.arg_size.name )
            controller.modify_arg_expression( self.arg_size_index, l_arr )
            doc_dependent(self.function, self.arg_size, self.arg)

        # dealing with output arrays
        for key, scale in self.output_arrays:
            oo_arg = self.get_argument(key)            
            oo_idx = self.function.arguments.index(oo_arg)
            oo_elem_pds = common.unique_pds(get_array_item_type(oo_arg.type).partial_decl_string)
            oo_vec_pds = get_vector_pds(oo_elem_pds)
            oa_init = ("(%s)" % l_arr) if scale=='1' else ("(%s*%s)" % (l_arr, scale))
            oa_arg = controller.declare_variable(_D.dummy_type_t(oo_vec_pds), key, oa_init)
            doc_common(self.function, oo_arg, common.current_sb.get_registered_decl_name(oo_vec_pds))
            controller.modify_arg_expression(oo_idx, "(%s)(&%s[0])" \
                % (oo_arg.type.partial_decl_string, oa_arg))
            controller.remove_wrapper_arg(key)

            controller.return_variable(oa_arg)
            doc_output(self.function, oo_arg)
        
            
    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_array1d( *args, **keywd ):
    def creator( function ):
        return input_array1d_t( function, *args, **keywd )
    return creator
    
    
# input_as_list_of_Matlike_t
class input_as_list_of_Matlike_t(transformer_t):
    """Converts an input/inout array argument into a std::vector<cv::Mat or cv::MatND>."""

    mat_c2cpp = { # [conversion func, output type]
        "IplImage": ["(IplImage)", "cv::Mat"],
        "IplImage *": ["get_IplImage_ptr", "cv::Mat"],
        "IplImage const *": ["get_IplImage_ptr", "cv::Mat"],
        "CvMat": ["(CvMat)", "cv::Mat"],
        "CvMat *": ["get_CvMat_ptr", "cv::Mat"],
        "CvMat const *": ["get_CvMat_ptr", "cv::Mat"],
        "void *": ["get_CvMat_ptr", "cv::Mat"],
        "void const *": ["get_CvMat_ptr", "cv::Mat"],
        "CvArr *": ["get_CvMat_ptr", "cv::Mat"],
        "CvArr const *": ["get_CvMat_ptr", "cv::Mat"],
        "cv::Mat *": ["&", "cv::Mat"],
        "cv::Mat const *": ["&", "cv::Mat"],
        "CvMatND": ["(CvMatND)", "cv::MatND"],
        "CvMatND *": ["get_CvMatND_ptr", "cv::MatND"],
        "CvMatND const *": ["get_CvMatND_ptr", "cv::MatND"],
        "cv::MatND *": ["&", "cv::MatND"],
        "cv::MatND const *": ["&", "cv::MatND"],
    }
    
    def __init__(self, function, arg_ref, arg_size_ref=None):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        self.arg_pds = common.unique_pds(self.arg.type.partial_decl_string)
        
        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
        else:
            self.arg_size = None
            
        self.in_elem_pds = common.unique_pds(get_array_item_type(self.arg.type).partial_decl_string)
        self.conversion_func, self.out_elem_pds = self.mat_c2cpp[self.in_elem_pds] 
        self.vec_pds = get_vector_pds(self.out_elem_pds)

    def __str__(self):
        return "input_as_list_of_Matlike(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)
        w_arg.type = _D.dummy_type_t(self.vec_pds+" const &")
        
        # what about non-NULL default value?
        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = self.vec_pds+'()'
        
        # pre_call
        controller.add_pre_call_code('''
    std::vector<IN_ELEM_TYPE> buf_ARG(ARG.size());
    for(size_t i_ARG = 0; i_ARG<ARG.size(); ++i_ARG)
        buf_ARG[i_ARG] = (IN_ELEM_TYPE)CONVERSION_FUNC(ARG[i_ARG]);
        '''.replace("ARG", w_arg.name).replace("IN_ELEM_TYPE", self.in_elem_pds)\
        .replace("CONVERSION_FUNC", self.conversion_func))
        
        # main code
        controller.modify_arg_expression( self.arg_index, "(%s)(&buf_%s[0])" \
            % (self.arg_pds, w_arg.name) )
            
        # remove arg_size from the function wrapper definition and automatically fill in the missing argument
        if self.arg_size is not None:
            controller.remove_wrapper_arg( self.arg_size.name )
            controller.modify_arg_expression( self.arg_size_index, w_arg.name+".size()" )
            doc_dependent(self.function, self.arg_size, self.arg)

        # documentation
        doc_common(self.function, self.arg, common.current_sb.get_registered_decl_name(self.vec_pds))

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
        return ["opencv_converters.hpp"]

def input_as_list_of_Matlike( *args, **keywd ):
    def creator( function ):
        return input_as_list_of_Matlike_t( function, *args, **keywd )
    return creator    


# input_list_of_string_t
class input_list_of_string_t(transformer.transformer_t):
    """Handles an input array of strings with a dynamic size.

    void do_something([int N, ]char [const ]* v) ->  do_something(object v2)

    where v2 is a bp::list of string.
    """

    def __init__(self, function, arg_ref, arg_size_ref=None, remove_arg_size=True):
        transformer.transformer_t.__init__( self, function )

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _T.is_ptr_or_array( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_list_of_string" transformation, argument %s type must be a array or a pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)

        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
            
            if not _D.is_integral( self.arg_size.type ):
                raise ValueError( '%s\nin order to use "input_list_of_string" transformation, argument %s type must be an integer (got %s).' ) \
                      % ( function, self.arg_size.name, self.arg_size.type)

        else:
            self.arg_size = None

        self.remove_arg_size = remove_arg_size

    def __str__(self):
        if self.arg_size is not None:
            return "input_list_of_string(%s,%s)"%( self.arg.name, self.arg_size.name)
        return "input_list_of_string(%s)"% self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["opencv_converters.hpp"]

    def __configure_sealed(self, controller):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        w_arg.type = _D.dummy_type_t('bp::list const &')
        
        # what about non-NULL default value?
        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = 'bp::list()'
            
        # number of elements
        l_arr = controller.declare_variable( _D.dummy_type_t('int'), self.arg.name, "=bp::len(%s)" % w_arg.name )
        if self.remove_arg_size and self.arg_size is not None:
            # remove arg_size from the function wrapper definition and automatically fill in the missing argument
            controller.remove_wrapper_arg( self.arg_size.name )
            controller.modify_arg_expression( self.arg_size_index, l_arr )
            doc_dependent(self.function, self.arg_size, self.arg)

        # intermediate array
        a_arr = controller.declare_variable(_D.dummy_type_t('std::vector<char const *>'), self.arg.name)
        controller.add_pre_call_code('A_ARR.resize(L_ARR); while(--L_ARR >= 0) A_ARR[L_ARR]=bp::extract<char const *>(ARRAY[L_ARR]);'\
            .replace('L_ARR', l_arr).replace('A_ARR', a_arr).replace('ARRAY', w_arg.name))
        
        # input array
        controller.modify_arg_expression( self.arg_index, "(%s)(&%s[0])" \
            % (self.arg.type.partial_decl_string, a_arr) )
        doc_common(self.function, self.arg, "list of strings")

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_list_of_string( *args, **keywd ):
    def creator( function ):
        return input_list_of_string_t( function, *args, **keywd )
    return creator


class input_array2d_t(transformer.transformer_t):
    """Handles an input array with a dynamic size.

    void do_something([[int N, ]int *ncnts, ]data_type** v=NULL) ->  do_something(object v2)

    where v2 is a Python sequence of sequences of items, each of which is of type 'data_type'.
    Note that if 'data_type' is replaced by 'CvSomething *', each element of v2 is still of type 'CvSomething' (i.e. the pointer is taken care of).
    If v2 is None, then v=ncnts=NULL and N=0.
    """

    def __init__(self, function, arg_ref, arg_ncnts_ref=None, arg_size_ref=None, remove_arg_ncnts=True, remove_arg_size=True):
        transformer.transformer_t.__init__( self, function )

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _T.is_ptr_or_array( self.arg.type ) or not _T.is_ptr_or_array(remove_ptr(self.arg.type)):
            raise ValueError( '%s\nin order to use "input_array2d" transformation, argument %s type must be a double array or a double pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)
                  
        if arg_ncnts_ref is not None:
            self.arg_ncnts = self.get_argument( arg_ncnts_ref )
            self.arg_ncnts_index = self.function.arguments.index( self.arg_ncnts )
            
            if not _T.is_ptr_or_array(self.arg_ncnts.type) or not _D.is_integral( remove_ptr(self.arg_ncnts.type) ):
                raise ValueError( '%s\nin order to use "input_array2d" transformation, argument %s type must be an integer array (got %s).' ) \
                      % ( function, self.arg_ncnts.name, self.arg_ncnts.type)

        else:
            self.arg_ncnts = None
        self.remove_arg_ncnts = remove_arg_ncnts

        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
            
            if not _D.is_integral( self.arg_size.type ):
                raise ValueError( '%s\nin order to use "input_array2d" transformation, argument %s type must be an integer (got %s).' ) \
                      % ( function, self.arg_size.name, self.arg_size.type)

        else:
            self.arg_size = None
        self.remove_arg_size = remove_arg_size

        self.array_item_type = get_array_item_type(get_array_item_type(self.arg.type))

    def __str__(self):
        return "input_array2d(%s)"% (self.arg.name,
            "None" if self.arg_ncnts is None else self.arg_ncnts.name,
            "None" if self.arg_sizes is None else self.arg_sizes.name,
            )

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ code_repository.convenience.file_name, "opencv_converters.hpp" ]

    def __configure_sealed(self, controller):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        elem_pds = common.unique_pds(self.array_item_type.partial_decl_string)
        vec_pds = get_vector_pds(get_vector_pds(elem_pds))
        w_arg.type = _D.dummy_type_t(vec_pds+" const &")

        if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
            w_arg.default_value = vec_pds+"()"
        doc_common(self.function, self.arg, common.current_sb.get_registered_decl_name(vec_pds))
        
        if self.remove_arg_size and self.arg_size is not None:
            #removing arg_size from the function wrapper definition
            controller.remove_wrapper_arg( self.arg_size.name )
            doc_dependent(self.function, self.arg_size, self.arg)
            
        if self.remove_arg_ncnts and self.arg_ncnts is not None:
            #removing arg_ncnts from the function wrapper definition
            controller.remove_wrapper_arg( self.arg_ncnts.name )
            doc_dependent(self.function, self.arg_ncnts, self.arg)

        # precall_code
        precall_code = """size_t n0_ARRAY = ARRAY.size();
    std::vector< ITEM_TYPE * > buf_ARRAY(n0_ARRAY);
    std::vector<size_t> n1_ARRAY(n0_ARRAY);
    for(size_t i_ARRAY = 0; i_ARRAY<n0_ARRAY; ++i_ARRAY)
    {
        buf_ARRAY[i_ARRAY] = (ITEM_TYPE *)(&ARRAY[i_ARRAY][0]);
        n1_ARRAY[i_ARRAY] = ARRAY[i_ARRAY].size();
    }
        """.replace("ARRAY", self.arg.name).replace("ITEM_TYPE", elem_pds)
        controller.add_pre_call_code(precall_code)
        
        controller.modify_arg_expression( self.arg_index, "(%s)(&buf_%s[0])" \
            % (self.arg.type.partial_decl_string, self.arg.name) )
        
        if self.remove_arg_ncnts and self.arg_ncnts is not None:
            controller.modify_arg_expression( self.arg_ncnts_index, "(%s)(&n1_%s[0])" 
                % (self.arg_ncnts.type.partial_decl_string, self.arg.name) )

        if self.remove_arg_size and self.arg_size is not None:
            controller.modify_arg_expression( self.arg_size_index, "n0_"+self.arg.name )

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_array2d( *args, **keywd ):
    def creator( function ):
        return input_array2d_t( function, *args, **keywd )
    return creator



# output_type1_t
class output_type1_t( transformer.transformer_t ):
    """Handles a single output variable.

    The specified variable is removed from the argument list and is turned
    into a return value.

    void getValue(data_type* v) -> v2 = getValue()

    where v2 is of type 'data_type'.
    Note that if 'data_type' is replaced by 'CvSomething *', v2 is still of type 'CvSomething' (i.e. the pointer is taken care of).
    And note that the value of *v is initialized to NULL (if it is a pointer) before v is passed to C function getValue().
    And by default, call policies are ignored.
    """

    def __init__(self, function, arg_ref, ignore_call_policies=True):
        transformer.transformer_t.__init__( self, function )
        """Constructor.

        The specified argument must be a reference or a pointer.

        :param arg_ref: Index of the argument that is an output value.
        :type arg_ref: int
        """
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        self.ignore_call_policies = ignore_call_policies

        if not _D.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "output_type1" transformation, argument %s type must be a pointer (got %s).' ) \
                  % ( function, self.arg.name, arg.type)

    def __str__(self):
        return "output_type1(%d)"%(self.arg.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ code_repository.convenience.file_name, "ndarray.hpp" ]

    def __configure_sealed( self, controller ):
        # removing arg from the function wrapper definition
        controller.remove_wrapper_arg( self.arg.name )
        # documentation
        pds = _D.remove_const(_D.remove_pointer(self.arg.type)).partial_decl_string
        doc_common(self.function, self.arg, common.current_sb.get_registered_decl_name(pds))
        doc_output(self.function, self.arg)
        #the element type
        if self.arg.type.partial_decl_string=='::size_t *':
            etype = _D.dummy_type_t('::size_t') 
        else:
            etype = _D.remove_pointer( self.arg.type ) 
        #declaring new variable, which will keep result
        if _D.is_pointer(etype):
            var_name = controller.declare_variable( etype, self.arg.name, "=(%s)0" % etype.decl_string )
        else:
            var_name = controller.declare_variable( etype, self.arg.name )
        #adding just declared variable to the original function call expression
        controller.modify_arg_expression( self.arg_index, "&" + var_name )
        #adding the variable to return variables list
        controller.return_variable( var_name if self.ignore_call_policies else 'pyplusplus::call_policies::make_object< call_policies_t, %s >( %s )' % (etype.decl_string, var_name) )

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def output_type1( *args, **keywd ):
    def creator( function ):
        return output_type1_t( function, *args, **keywd )
    return creator


# arg_output_t
class arg_output_t( transformer.transformer_t ):
    """Handles a single output variable.

    The specified variable is removed from the argument list and is turned
    into a return value.

    void getValue(data_type v) -> v2 = getValue()

    where v2 is of type 'data_type'.
    """

    def __init__(self, function, arg_ref, ignore_call_policies=True):
        transformer.transformer_t.__init__( self, function )
        """Constructor.

        The specified argument must be a reference or a pointer.

        :param arg_ref: Index of the argument that is an output value.
        :type arg_ref: int
        """
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        self.ignore_call_policies = ignore_call_policies

    def __str__(self):
        return "arg_output(%d)"%(self.arg.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return []

    def __configure_sealed( self, controller ):
        # removing arg from the function wrapper definition
        controller.remove_wrapper_arg( self.arg.name )
        # documentation
        elem_type = _D.remove_const(_D.remove_reference(self.arg.type))
        doc_common(self.function, self.arg, common.current_sb.get_registered_decl_name(elem_type.partial_decl_string))
        doc_output(self.function, self.arg)
        # declaring new variable, which will keep result
        elem_type = _D.dummy_type_t(common.unique_pds(elem_type.partial_decl_string))
        var_name = controller.declare_variable(elem_type, self.arg.name)
        # adding just declared variable to the original function call expression
        controller.modify_arg_expression( self.arg_index, var_name )
        # adding the variable to return variables list
        controller.return_variable(var_name)

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def arg_output( *args, **keywd ):
    def creator( function ):
        return arg_output_t( function, *args, **keywd )
    return creator


    
# trackbar_callback2_func_t
class trackbar_callback2_func_t(transformer.transformer_t):
    """Handles a CvTrackbarCallback argument.

    void do_something(CvTrackbarCallback on_change, void* param) ->  do_something((Python function) on_change, (object) param)
    """

    def __init__(self, function, arg_on_mouse, arg_user_data):
        transformer.transformer_t.__init__( self, function )

        self.arg1 = self.get_argument( arg_on_mouse )
        self.arg1_index = self.function.arguments.index( self.arg1 )

        self.arg2 = self.get_argument( arg_user_data )
        self.arg2_index = self.function.arguments.index( self.arg2 )


    def __str__(self):
        return "trackbar_callback2_func(%s,%s)"% (self.arg1.name, self.arg2.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["boost/python/object.hpp", "boost/python/tuple.hpp", "opencv_extra.hpp" ]

    def __configure_sealed(self, controller):
        w_arg1 = controller.find_wrapper_arg( self.arg1.name )
        w_arg1.type = _D.dummy_type_t( "boost::python::object" )

        w_arg2 = controller.find_wrapper_arg( self.arg2.name )
        w_arg2.type = _D.dummy_type_t( "boost::python::object" )

        if self.arg1.default_value == '0' or self.arg1.default_value == 'NULL':
            w_arg1.default_value = 'bp::object()'
        
        if self.arg2.default_value == '0' or self.arg2.default_value == 'NULL':
            w_arg2.default_value = 'bp::object()'
        
        # declare a tuple to keep the function and the parameter together
        var_tuple = controller.declare_variable( _D.dummy_type_t("bp::tuple"), "z_"+w_arg1.name,
            "= bp::make_tuple(%s, %s)" % (w_arg1.name, w_arg2.name))
        
        # adding the variable to return variables list
        # controller.return_variable( 'pyplusplus::call_policies::make_object< call_policies_t, bp::tuple >( %s )' % var_tuple )
        controller.return_variable( var_tuple )

        controller.modify_arg_expression( self.arg1_index, "sdTrackbarCallback2" )
        controller.modify_arg_expression( self.arg2_index, "(%s)(%s.ptr())" % (self.arg2.type.decl_string, var_tuple))
        
        # documentation
        common.add_decl_boost_doc(self.function, "Argument '%s' is a Python function that should look like below:" % self.arg1.name)
        common.add_decl_boost_doc(self.function, "    def on_trackbar(pos, user_data):")
        common.add_decl_boost_doc(self.function, "        ...")
        common.add_decl_boost_doc(self.function, "Argument '%s' is a Python object that is passed to function on_trackbar() as 'user_data'." % self.arg2.name)
        

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def trackbar_callback2_func( *args, **keywd ):
    def creator( function ):
        return trackbar_callback2_func_t( function, *args, **keywd )
    return creator


# mouse_callback_func_t
class mouse_callback_func_t(transformer.transformer_t):
    """Handles a CvMouseCallback argument.

    void do_something(CvMouseCallback on_mouse, void* param) ->  do_something((Python function) on_mouse, (object) param)
    """

    def __init__(self, function, arg_on_mouse, arg_param):
        transformer.transformer_t.__init__( self, function )

        self.arg1 = self.get_argument( arg_on_mouse )
        self.arg1_index = self.function.arguments.index( self.arg1 )

        self.arg2 = self.get_argument( arg_param )
        self.arg2_index = self.function.arguments.index( self.arg2 )


    def __str__(self):
        return "mouse_callback_func(%s,%s)"% (self.arg1.name, self.arg2.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["boost/python/object.hpp", "boost/python/tuple.hpp", "opencv_extra.hpp" ]

    def __configure_sealed(self, controller):
        w_arg1 = controller.find_wrapper_arg( self.arg1.name )
        w_arg1.type = _D.dummy_type_t( "boost::python::object" )

        w_arg2 = controller.find_wrapper_arg( self.arg2.name )
        w_arg2.type = _D.dummy_type_t( "boost::python::object" )

        if self.arg2.default_value == '0' or self.arg2.default_value == 'NULL':
            w_arg2.default_value = 'bp::object()'
        
        # declare a tuple to keep the function and the parameter together
        var_tuple = controller.declare_variable( _D.dummy_type_t("boost::python::tuple"), "z_"+w_arg1.name,
            "= bp::make_tuple(%s, %s)" % (w_arg1.name, w_arg2.name))
        
        # adding the variable to return variables list
        # controller.return_variable( 'pyplusplus::call_policies::make_object< call_policies_t, bp::tuple >( %s )' % var_tuple )
        controller.return_variable( var_tuple )

        controller.modify_arg_expression( self.arg1_index, "sdMouseCallback" )
        controller.modify_arg_expression( self.arg2_index, "(%s)(%s.ptr())" % (self.arg2.type.decl_string, var_tuple))
        
        # documentation
        common.add_decl_boost_doc(self.function, "Argument '%s' is a Python function that should look like below:" % self.arg1.name)
        common.add_decl_boost_doc(self.function, "    def on_mouse(event, x, y, flags, user_data):")
        common.add_decl_boost_doc(self.function, "        ...")
        common.add_decl_boost_doc(self.function, "Argument '%s' is a Python object that is passed to function on_mouse() as 'user_data'." % self.arg2.name)
        

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def mouse_callback_func( *args, **keywd ):
    def creator( function ):
        return mouse_callback_func_t( function, *args, **keywd )
    return creator


# distance_function_t
class distance_function_t(transformer.transformer_t):
    """Handles a CvMouseCallback argument.

    void do_something(CvMouseCallback on_mouse, void* param) ->  do_something((Python function) on_mouse, (object) param)
    """

    def __init__(self, function, arg_distance_func, arg_userdata):
        transformer.transformer_t.__init__( self, function )

        self.arg1 = self.get_argument( arg_distance_func )
        self.arg1_index = self.function.arguments.index( self.arg1 )

        self.arg2 = self.get_argument( arg_userdata )
        self.arg2_index = self.function.arguments.index( self.arg2 )


    def __str__(self):
        return "distance_function(%s,%s)"% (self.arg1.name, self.arg2.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["boost/python/object.hpp", "boost/python/tuple.hpp" ]

    def __configure_sealed(self, controller):
        w_arg1 = controller.find_wrapper_arg( self.arg1.name )
        w_arg1.type = _D.dummy_type_t( "boost::python::object" )

        if self.arg1.default_value == '0' or self.arg1.default_value == 'NULL':
            w_arg1.default_value = 'bp::object()'

        w_arg2 = controller.find_wrapper_arg( self.arg2.name )
        w_arg2.type = _D.dummy_type_t( "boost::python::object" )

        if self.arg2.default_value == '0' or self.arg2.default_value == 'NULL':
            w_arg2.default_value = 'bp::object()'

        # declare a variable to check if distance_func is None
        b_dist_func = controller.declare_variable( _D.dummy_type_t("bool"), "b_"+w_arg1.name, "= %s.ptr() != Py_None" % w_arg1.name)
        
        # declare a tuple to keep the function and the parameter together
        var_tuple = controller.declare_variable( _D.dummy_type_t("boost::python::tuple"), "z_"+w_arg1.name)
        
        # precall code
        controller.add_pre_call_code("if(%s) %s = bp::make_tuple(%s, %s);" % (b_dist_func, var_tuple, w_arg1.name, w_arg2.name))
        
        controller.modify_arg_expression( self.arg1_index, "%s? sdDistanceFunction: 0" % b_dist_func )
        controller.modify_arg_expression( self.arg2_index, "%s? (%s)(%s.ptr()): 0" % (b_dist_func, self.arg2.type.decl_string, var_tuple))
        

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def distance_function( *args, **keywd ):
    def creator( function ):
        return distance_function_t( function, *args, **keywd )
    return creator


# input_as_Seq_t
class input_as_Seq_t(transformer.transformer_t):
    """Handles an input of type CvSeq as input of type Seq<T>.

    void do_something(CvSeq* v[, CvMemStorage *ms]) ->  do_something(Seq<T> v2)
    void do_something(CvSeq** v[, CvMemStorage *ms]) ->  do_something(Seq<T> v2)

    where v2 is a Seq of type T.
    """

    def __init__(self, function, elem_type_pds, arg_ref, arg_size_ref=None, remove_arg_size=True):
        transformer.transformer_t.__init__( self, function )
        
        self.elem_type_pds = elem_type_pds

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _D.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_as_Seq" transformation, argument %s type must be a pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)

        if arg_size_ref is not None:
            self.arg_size = self.get_argument( arg_size_ref )
            self.arg_size_index = self.function.arguments.index( self.arg_size )
            
            if not 'CvMemStorage' in self.arg_size.type.partial_decl_string:
                raise ValueError( '%s\nin order to use "input_as_Seq" transformation, argument %s type must be a CvMemStorage (got %s).' ) \
                      % ( function, self.arg_size.name, self.arg_size.type)

        else:
            self.arg_size = None

        self.remove_arg_size = remove_arg_size

    def __str__(self):
        if self.arg_size is not None:
            return "input_as_Seq(%s, %s,%s)"%( self.elem_type_pds, self.arg.name, self.arg_size.name)
        return "input_as_Seq(%s, %s)"% (self.elem_type_pds, self.arg.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["opencv_converters.hpp"]

    def __configure_sealed(self, controller):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        seq_pds = common.unique_pds("cv::Seq< %s > &" % self.elem_type_pds)
        w_arg.type = _D.dummy_type_t(seq_pds)
        doc_common(self.function, self.arg, common.current_sb.get_decl_equivname(seq_pds[:-2]))
        
        if '* *' in self.arg.type.partial_decl_string:
            controller.modify_arg_expression(self.arg_index, "&%s.seq" % w_arg.name)
        else:
            controller.modify_arg_expression(self.arg_index, "%s.seq" % w_arg.name)
        
        # memory storage
        if self.remove_arg_size and self.arg_size is not None:
            # remove arg_size from the function wrapper definition and automatically fill in the missing argument
            controller.remove_wrapper_arg( self.arg_size.name )
            controller.modify_arg_expression( self.arg_size_index, "%s.seq->storage" % w_arg.name)
            doc_dependent(self.function, self.arg_size, self.arg)
            
    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_as_Seq( *args, **keywd ):
    def creator( function ):
        return input_as_Seq_t( function, *args, **keywd )
    return creator


# input_CvSeq_ptr_as_vector_t
class input_CvSeq_ptr_as_vector_t(transformer.transformer_t):
    """Handles an input of type CvSeq* as input of type std::vector<T>.

    void do_something(CvSeq* v) ->  do_something(std::vector<T> v2)
    """

    def __init__(self, function, elem_type_pds, arg_ref):
        transformer.transformer_t.__init__( self, function )
        
        self.elem_type_pds = elem_type_pds

        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

        if not _D.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_CvSeq_ptr_as_vector" transformation, argument %s type must be a pointer (got %s).' ) \
                  % ( function, self.arg.name, self.arg.type)


    def __str__(self):
        return "input_CvSeq_ptr_as_vector(%s, %s)"% (self.elem_type_pds, self.arg.name)

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return ["opencv_converters.hpp"]

    def __configure_sealed(self, controller):
        controller.remove_wrapper_arg( self.arg.name )
        vec_pds = common.unique_pds("std::vector< %s >" % self.elem_type_pds)
        doc_common(self.function, self.arg, "vector_float32")
        doc_output(self.function, self.arg)                    
        seq = controller.declare_variable(_D.dummy_type_t("CvSeq *"), self.arg.name, "=0" )
        controller.modify_arg_expression(self.arg_index, "&"+seq)
        controller.return_variable("convert_CvSeq_ptr_to_vector<%s>(%s)" % (self.elem_type_pds, seq))

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

def input_CvSeq_ptr_as_vector( *args, **keywd ):
    def creator( function ):
        return input_CvSeq_ptr_as_vector_t( function, *args, **keywd )
    return creator



    
# input_asSparseMat_t
class input_asSparseMat_t(transformer_t):
    """Converts an input argument type into a cv::SparseMat."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_asSparseMat(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)        
        w_arg.type = _D.dummy_type_t( "::cv::SparseMat &" )
        dtype = self.arg.type
        
        if dtype == _D.dummy_type_t("::CvSparseMat *"):
            controller.modify_arg_expression( self.arg_index, "&%s.state" % w_arg.name )
        elif dtype == _D.dummy_type_t("::CvSparseMat &") or dtype == _D.dummy_type_t("::CvSparseMat"):
            controller.modify_arg_expression( self.arg_index, "%s.state" % w_arg.name )
        else:
            raise NotImplementedError("Input argument type %s is not convertible into cv::SparseMat." % dtype.decl_string)
            
        # documentation
        doc_common(self.function, self.arg, "SparseMat")

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

def input_asSparseMat( *args, **keywd ):
    def creator( function ):
        return input_asSparseMat_t( function, *args, **keywd )
    return creator
    


# input_as_MemStorage_t
class input_as_MemStorage_t(transformer_t):
    """Converts an input argument type CvMemStorage * into a cv::MemStorage."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_as_MemStorage(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)        
        w_arg.type = _D.dummy_type_t( "::cv::MemStorage &" )
        
        if self.arg.default_value is not None:
            w_arg.default_value = "cv::MemStorage(%s)" % self.arg.default_value
            w_arg.type = _D.dummy_type_t( "::cv::MemStorage" )
        else:
            w_arg.type = _D.dummy_type_t( "::cv::MemStorage &" )
        
        controller.modify_arg_expression( self.arg_index, "(CvMemStorage *)"+w_arg.name )
            
        # documentation
        doc_common(self.function, self.arg, "MemStorage")


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

def input_as_MemStorage( *args, **keywd ):
    def creator( function ):
        return input_as_MemStorage_t( function, *args, **keywd )
    return creator
    
    
# input_as_FileStorage_t
class input_as_FileStorage_t(transformer_t):
    """Converts an input argument type CvFileStorage * into a cv::FileStorage."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_as_FileStorage(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)        
        w_arg.type = _D.dummy_type_t( "::cv::FileStorage &" )
        
        controller.modify_arg_expression( self.arg_index, "%s.fs" % w_arg.name )
            
        # documentation
        doc_common(self.function, self.arg, "FileStorage")


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

def input_as_FileStorage( *args, **keywd ):
    def creator( function ):
        return input_as_FileStorage_t( function, *args, **keywd )
    return creator
    
    
# input_as_FileNode_t
class input_as_FileNode_t(transformer_t):
    """Converts an input argument type CvFileNode * into a cv::FileNode."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_as_FileNode(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)
        if self.arg.type == _D.dummy_type_t("::CvFileNode *"):
            w_arg.type = _D.dummy_type_t( "::cv::FileNode &" )
        else:
            w_arg.type = _D.dummy_type_t( "::cv::FileNode const &" )
        controller.modify_arg_expression( self.arg_index, "*(%s)" % w_arg.name )
            
        # documentation
        doc_common(self.function, self.arg, "FileNode")


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

def input_as_FileNode( *args, **keywd ):
    def creator( function ):
        return input_as_FileNode_t( function, *args, **keywd )
    return creator
    

# input_as_FixType_t
class input_as_FixType_t(transformer_t):
    """Converts an input argument type SrcType into a DstType, both are fixed-size data types."""

    def __init__(self, function, src_type_pds, dst_type_pds, arg_ref, by_ref=True):
        transformer.transformer_t.__init__( self, function )
        self.src_type_pds = src_type_pds
        self.dst_type_pds = dst_type_pds
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        self.by_ref = by_ref

    def __str__(self):
        return "input_as_FixType(%s,%s,%s)" % (self.src_type_pds, self.dst_type_pds, self.arg.name)

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)
        w_arg.type = _D.dummy_type_t(self.dst_type_pds+" const &")
        if self.arg.default_value is not None:
            s = self.arg.default_value
            repl_dict = {
                'cvPoint': 'cv::Point',
                'cvTermCriteria': 'cv::TermCriteria',
                'cvSlice(0, 1073741823)': 'cv::Range::all()',
            }
            for z in repl_dict:
                s = s.replace(z, repl_dict[z])
            w_arg.default_value = s
            
        if self.by_ref:
            if _D.is_reference(self.arg.type):
                src_pds = common.unique_pds(self.arg.type.partial_decl_string)
            else:
                src_pds = self.src_type_pds+" const &"
        else:
            src_pds = self.src_type_pds
        controller.modify_arg_expression( self.arg_index, "(%s)%s" % (src_pds, w_arg.name) )

        # documentation
        doc_common(self.function, self.arg, common.current_sb.get_decl_equivname(self.dst_type_pds))            

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

def input_as_FixType( *args, **keywd ):
    def creator( function ):
        return input_as_FixType_t( function, *args, **keywd )
    return creator
    
    
    
    
# input_asRNG_t
class input_asRNG_t(transformer_t):
    """Converts an input argument type into a cv::RNG."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_asRNG(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)        
        w_arg.type = _D.dummy_type_t( "::cv::RNG &" )
        dtype = self.arg.type
        
        if dtype == _D.dummy_type_t("::CvRNG *"):
            controller.modify_arg_expression( self.arg_index, "&%s.state" % w_arg.name )
        elif dtype == _D.dummy_type_t("::CvRNG &") or dtype == _D.dummy_type_t("::CvRNG"):
            controller.modify_arg_expression( self.arg_index, "%s.state" % w_arg.name )
        else:
            raise NotImplementedError("Input argument type %s is not convertible into cv::RNG." % dtype.decl_string)
            
        # documentation
        doc_common(self.function, self.arg, "RNG")


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

def input_asRNG( *args, **keywd ):
    def creator( function ):
        return input_asRNG_t( function, *args, **keywd )
    return creator
    
    
# input_as_Mat_t
class input_as_Mat_t(transformer_t):
    """Converts an input/inout argument type into a cv::Mat."""

    def __init__(self, function, arg_ref):
        transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_as_Mat(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg(self.arg.name)
        dtype = self.arg.type
        if dtype == _D.dummy_type_t("::IplImage *") \
            or dtype == _D.dummy_type_t("::IplImage const *"):
            
            # default value
            if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
                w_arg.type = _D.dummy_type_t( "::cv::Mat" )
                w_arg.default_value = 'cv::Mat()'
            else:
                w_arg.type = _D.dummy_type_t( "::cv::Mat &" )
                
            # code
            controller.modify_arg_expression( self.arg_index, "get_IplImage_ptr(%s)" % w_arg.name)
                
            # documentation
            doc_Mat(self.function, self.arg)

        elif dtype == _D.dummy_type_t("::CvMat *") \
            or dtype == _D.dummy_type_t("::CvMat const *") \
            or dtype == _D.dummy_type_t("::CvArr *") \
            or dtype == _D.dummy_type_t("::CvArr const *"):
            
            # default value
            if self.arg.default_value == '0' or self.arg.default_value == 'NULL':
                w_arg.type = _D.dummy_type_t( "::cv::Mat" )
                w_arg.default_value = 'cv::Mat()'
            else:
                w_arg.type = _D.dummy_type_t( "::cv::Mat &" )
                
            # code
            controller.modify_arg_expression( self.arg_index, "get_CvMat_ptr(%s)" % w_arg.name)

            # documentation
            doc_Mat(self.function, self.arg)

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
        return ["opencv_converters.hpp"]

def input_as_Mat( *args, **keywd ):
    def creator( function ):
        return input_as_Mat_t( function, *args, **keywd )
    return creator
    
    
