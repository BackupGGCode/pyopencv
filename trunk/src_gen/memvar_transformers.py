#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.0 using Boost.Python and NumPy

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
import pyplusplus.function_transformers.transformers as _T
from pyplusplus.decl_wrappers import call_policies as CP

import common


# -----------------------------------------------------------------------------------------------
# Some functions
# -----------------------------------------------------------------------------------------------


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
    
def expose_member_as_Mat(klass, member_name, is_CvMat_ptr=True):
    klass.var(member_name).exclude()
    CvMat = 'CvMat' if is_CvMat_ptr else 'IplImage'
    klass.add_wrapper_code('''
    cv::Mat MEMBER_NAME_as_Mat;
    CVMAT MEMBER_NAME_as_CvMat;
    void set_MEMBER_NAME(cv::Mat const &new_MEMBER_NAME)
    {
        MEMBER_NAME_as_Mat = new_MEMBER_NAME; // to keep a reference to MEMBER_NAME
        MEMBER_NAME_as_CVMAT = MEMBER_NAME_as_Mat; // to ensure MEMBER_NAME points to a valid CVMAT
        MEMBER_NAME = &MEMBER_NAME_as_CVMAT;
    }
    cv::Mat & get_MEMBER_NAME()
    {
        if(MEMBER_NAME != &MEMBER_NAME_as_CVMAT) set_MEMBER_NAME(cv::Mat(MEMBER_NAME));
        return MEMBER_NAME_as_Mat;
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string).replace("CVMAT", CvMat))
    klass.add_registration_code('''add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME, bp::return_internal_reference<>()),
        &CLASS_TYPE_wrapper::set_MEMBER_NAME)'''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    
def expose_member_as_TermCriteria(klass, member_name):
    klass.var(member_name).exclude()
    klass.add_wrapper_code('''
    cv::TermCriteria get_MEMBER_NAME() { return cv::TermCriteria(MEMBER_NAME); }
    void set_MEMBER_NAME(cv::TermCriteria const &_MEMBER_NAME) { MEMBER_NAME = _MEMBER_NAME; }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('add_property( "MEMBER_NAME", &CLASS_TYPE_wrapper::get_MEMBER_NAME, &CLASS_TYPE_wrapper::set_MEMBER_NAME)' \
        .replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    
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
    z = klass.var(member_name)
    z.exclude()
    klass.add_declaration_code("static MEMBER_TYPE get_MEMBER_NAME( CLASS_TYPE const & inst ) { return inst.MEMBER_NAME; }"\
        .replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string)\
        .replace("MEMBER_TYPE", z.type.decl_string))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&::get_MEMBER_NAME, bp::return_internal_reference<>()) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    
def expose_member_as_array_of_pointees(klass, member_name, array_size):
    klass.include_files.append( "boost/python/object.hpp")
    klass.include_files.append( "boost/python/list.hpp")
    klass.include_files.append( "boost/python/tuple.hpp")
    klass.var(member_name).exclude() # TODO: with_custodian_and_ward for each pointee of the array
    klass.add_wrapper_code('''
    static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){
        bp::list l;
        for(int i = 0; i < ARRAY_SIZE; ++i)
            l.append(inst.MEMBER_NAME[i]);
        return bp::tuple(l);
    }
    '''.replace("MEMBER_NAME", member_name) \
        .replace("CLASS_TYPE", klass.decl_string) \
        .replace("ARRAY_SIZE", str(array_size)))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    
