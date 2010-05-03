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
import pyplusplus.function_transformers.transformers as _T
from pyplusplus.decl_wrappers import call_policies as CP

import common


# -----------------------------------------------------------------------------------------------
# Some functions
# -----------------------------------------------------------------------------------------------

class size_t_t( _D.fundamental_t ):
    """represents size_t type"""
    CPPNAME = 'size_t'
    def __init__( self ):
        _D.fundamental_t.__init__( self, size_t_t.CPPNAME )


def set_array_item_type_as_size_t(klass, member_name):
    _D.remove_cv(_D.remove_alias(klass.var(member_name).type)).base = size_t_t()

def expose_member_as_MemStorage(klass, member_name):
    klass.var(member_name).exclude()
    klass.add_declaration_code('''
static cv::MemStorage get_MEMBER_NAME(CLASS_TYPE const &inst) { return cv::MemStorage(inst.MEMBER_NAME); }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.pds))
    klass.add_registration_code('''add_property( "MEMBER_NAME", bp::make_function(&::get_MEMBER_NAME, bp::with_custodian_and_ward_postcall<0, 1>()) )'''.replace("MEMBER_NAME", member_name))
        
def expose_member_as_FixType(dst_type_pds, klass, member_name):
    klass.var(member_name).exclude()
    klass.add_declaration_code('''
static DST_TYPE *get_MEMBER_NAME(CLASS_TYPE const &inst) { return (DST_TYPE *)(&inst.MEMBER_NAME); }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.pds).replace("DST_TYPE", dst_type_pds))
    klass.add_registration_code('''add_property( "MEMBER_NAME", bp::make_function(&::get_MEMBER_NAME, bp::return_internal_reference<>()) )'''\
        .replace("MEMBER_NAME", member_name))
        
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
        
def expose_array_member_as_Mat(klass, member_name, member_size_name, extra="0"):
    klass.include_files.append( "opencv_converters.hpp" )
    klass.var(member_name).exclude()
    # klass.var(member_size_name).exclude()
    klass.add_declaration_code('''
static cv::Mat get_MEMBER_NAME(CLASS_TYPE const &inst)
{
    cv::Mat MEMBER_NAME2;
    convert_from_array_of_T_to_Mat(inst.MEMBER_NAME, inst.MEMBER_SIZE_NAME+EXTRA, MEMBER_NAME2);
    return MEMBER_NAME2;
}

    '''.replace("MEMBER_NAME", member_name).replace("MEMBER_SIZE_NAME", member_size_name).replace("CLASS_TYPE", klass.decl_string).replace("EXTRA", extra))
    klass.add_registration_code('''add_property( "MEMBER_NAME", &::get_MEMBER_NAME)'''.replace("MEMBER_NAME", member_name)) # TODO: make MEMBER dependent on KLASS
    
def expose_member_as_str(klass, member_name):
    klass.include_files.append( "boost/python/object.hpp" )
    klass.include_files.append( "boost/python/str.hpp" )
    klass.var(member_name).exclude()
    klass.add_declaration_code('''
static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){        
    return inst.MEMBER_NAME? bp::str(inst.MEMBER_NAME): bp::object();
}

    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('add_property( "MEMBER_NAME", &::get_MEMBER_NAME )' \
        .replace("MEMBER_NAME", member_name))
    
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
    klass.add_declaration_code('''
static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){
    bp::list l;
    for(int i = 0; i < ARRAY_SIZE; ++i)
        l.append(inst.MEMBER_NAME[i]);
    return bp::tuple(l);
}

    '''.replace("MEMBER_NAME", member_name) \
        .replace("CLASS_TYPE", klass.partial_decl_string) \
        .replace("ARRAY_SIZE", str(array_size)))
    klass.add_registration_code('add_property( "MEMBER_NAME", &::get_MEMBER_NAME )' \
        .replace("MEMBER_NAME", member_name))
    

    
# -----------------------------------------------------------------------------------------------
# Beautify all member variables of a class
# -----------------------------------------------------------------------------------------------
def beautify_memvars(klass):
    try:
        zz = klass.vars()
    except RuntimeError:
        zz = []
    
    for z in [z for z in zz if not z.ignore and z.access_type=='public']:
        pds = common.unique_pds(z.type.partial_decl_string)
        if pds=='CvMemStorage *':
            expose_member_as_MemStorage(klass, z.name)
        elif pds=='CvMat *' or pds=='CvArr *' or pds=='CvMat const *':
            expose_member_as_Mat(klass, z.name, True)
        elif pds=='IplImage *':
            expose_member_as_Mat(klass, z.name, False)
        elif pds in common.c2cpp:
            expose_member_as_FixType(common.c2cpp[pds], klass, z.name)
        elif pds=='CvSeq *' or pds=='CvSet *':
            expose_member_as_pointee(klass, z.name)
