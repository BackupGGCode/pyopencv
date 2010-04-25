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

from os import chdir, getcwd
import os.path as OP
import sys
from pygccxml import declarations as D
from pyplusplus import module_builder, messages
import function_transformers as FT
from pyplusplus.module_builder import call_policies as CP
from shutil import copyfile

# -----------------------------------------------------------------------------------------------
# modify pyplusplus.file_writers.multiple_files_t.split_creators to allow splitting into multiple files
# -----------------------------------------------------------------------------------------------
def my_split_creators( self, creators, pattern, function_name, registrator_pos, 
    prefix_level=0, n_creators=30 ):
    """Write non-class creators into multiple particular .h/.cpp files -- modified by Minh-Tri Pham.

    :param creators: The code creators that should be written
    :type creators: list of :class:`code_creators.code_creator_t`

    :param pattern: Name pattern that is used for constructing the final output file name
    :type pattern: str

    :param function_name: "register" function name
    :type function_name: str

    :param registrator_pos: The position of the code creator that creates the code to invoke the "register" function.
    :type registrator_pos: int

    :param n_creators: The number of code creators per file. -- Minh-Tri
    :type n_creators: int

    :param prefix_level: The current prefix level -- for internal use only. -- Minh-Tri
    :type prefix_level: int
    """
    
    def get_alias(creator):
        try:
            return creator.alias.lower()
        except AttributeError:
            return ''
    
    if len(creators) > n_creators:
        # get prefix characters
        charset = set()
        for creator in creators:
            creator_name = get_alias(creator)
            if len(creator_name) > prefix_level:
                charset.add(creator_name[prefix_level])
                
        # for each prefix character
        for char in charset:
            c1 = []
            c2 = []
            # split creators
            for creator in creators:
                creator_name = get_alias(creator)
                if len(creator_name) > prefix_level and creator_name[prefix_level]==char:
                    c1.append(creator)
                else:
                    c2.append(creator)
            creators = c2
            
            if prefix_level==0:
                self.split_creators(c1, pattern+'_'+char, function_name+'_'+char, 
                    registrator_pos, prefix_level+1, n_creators)
            else:
                self.split_creators(c1, pattern+char, function_name+char, 
                    registrator_pos, prefix_level+1, n_creators)
    self.split_creators_old(creators, pattern, function_name, registrator_pos)
    
import pyplusplus.file_writers as pf
pf.multiple_files_t.split_creators_old = pf.multiple_files_t.split_creators
pf.multiple_files_t.split_creators = my_split_creators

import cxerror_h
import cxtypes_h
import cxcore_h
import cxcore_hpp
import cxoperations_hpp
import cxflann_h
import cxmat_hpp
import cvtypes_h
import cv_h
import cv_hpp
import cvcompat_h
import cvaux_h
import cvaux_hpp
import cvvidsurv_hpp
import highgui_h
import highgui_hpp
import ml_h
import sdopencv

import common

_cwd = getcwd()
chdir(OP.join(OP.split(OP.abspath(__file__))[0], '..', 'src', 'pyopencv'))
_work_dir = getcwd()
print("Working directory changed to: %s" % _work_dir)

# -----------------------------------------------------------------------------------------------
# Creating an instance of class that will help you to expose your declarations
# -----------------------------------------------------------------------------------------------
mb = module_builder.module_builder_t(
    ["opencv_headers.hpp"],
    gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe",
    working_directory=OP.join(_work_dir, 'pyopencvext', 'core'),
    include_paths=[
        "pyopencvext/sdopencv",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
    ],
    )
common.mb = mb # register mb


# ===============================================================================================
# Start working
# ===============================================================================================

cc = open('core.py', 'w')
cc.write('''#!/usr/bin/env python
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
"""PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

Copyright (c) 2009, Minh-Tri Pham
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
   * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
"""

# Try to import numpy
try:
    import numpy as _NP
except ImportError:
    raise ImportError("NumPy is not found in your system. Please install NumPy of version at least 1.2.0.")

if _NP.version.version < '1.2.0':
    raise ImportError("NumPy is installed but its version is too old (%s detected). Please install NumPy of version at least 1.2.0." % _NP.version.version)


# Try to import pyopencvext
import config as _C
if _C.path_ext:
    import os as _os
    _seperator = ';' if _os.name == 'nt' else ':'
    _old_sys_path = _os.environ['PATH']
    _sys_path = _old_sys_path
    import config as _C
    for x in _C.path_ext:
        _sys_path = x + _seperator + _sys_path
    _os.environ['PATH'] = _sys_path
    # print("New path=",_sys_path)
    from pyopencvext import *
    import pyopencvext as _PE
    _os.environ['PATH'] = _old_sys_path
else:
    from pyopencvext import *
    import pyopencvext as _PE


import math as _Math
import ctypes as _CT


#=============================================================================
# cvver.h
#=============================================================================

CV_MAJOR_VERSION    = 2
CV_MINOR_VERSION    = 1
CV_SUBMINOR_VERSION = 0
CV_VERSION          = "2.1.0"




''')
mb.cc = cc

# -----------------------------------------------------------------------------------------------
# Subroutines related to writing to the core.py file
# -----------------------------------------------------------------------------------------------

def add_ndarray_interface(self, klass):
    klass.include_files.append("ndarray.hpp")
    klass.add_registration_code('def("from_ndarray", &sdcpp::from_ndarray< cv::%s >, (bp::arg("inst_ndarray")) )' % klass.alias)
    self.add_registration_code('bp::def("as%s", &sdcpp::from_ndarray< cv::%s >, (bp::arg("inst_ndarray")) );' % (klass.alias, klass.alias))
    klass.add_registration_code('staticmethod("from_ndarray")'.replace("KLASS", klass.alias))
    self.add_doc(klass.alias+".from_ndarray", "Creates a %s view on an ndarray instance." % klass.alias)
    klass.add_registration_code('add_property("ndarray", &sdcpp::as_ndarray< cv::%s >)' % klass.alias)
    # self.add_registration_code('bp::def("asndarray", &sdcpp::as_ndarray< cv::%s >, (bp::arg("inst_ndarray")) );' % klass.alias)
    self.add_doc(klass.alias,
        "Property 'ndarray' provides a numpy.ndarray view on the object.",
        "If you create a reference to 'ndarray', you must keep the object unchanged until your reference is deleted, or Python may crash!",
        # "Alternatively, you could create a reference to 'ndarray' by using 'asndarray(inst)', where 'inst' is an instance of this class.",
        "",
        "To create an instance of %s that shares the same data with an ndarray instance, use:" % klass.alias,
        "    '%s.from_ndarray(a)' or 'as%s(a)" % (klass.alias, klass.alias),
        "where 'a' is an ndarray instance. Similarly, to avoid a potential Python crash, you must keep the current instance unchanged until the reference is deleted.")
    for t in ('getitem', 'setitem', 'getslice', 'setslice', 'iter'):
        cc.write('''
def _KLASS__FUNC__(self, *args, **kwds):
    return self.ndarray.__FUNC__(*args, **kwds)
KLASS.__FUNC__ = _KLASS__FUNC__
        '''.replace('KLASS', klass.alias).replace('FUNC', t))
module_builder.module_builder_t.add_ndarray_interface = add_ndarray_interface

def expose_class_Ptr(self, klass_name, ns=None):
    if ns is None:
        full_klass_name = klass_name
    else:
        full_klass_name = '%s::%s' % (ns, klass_name)
    z = self.class_('Ptr<%s>' % full_klass_name)
    z.rename('Ptr_%s' % klass_name)
    z.include()
    z.operators().exclude()
    z.constructors(lambda x: '*' in x.decl_string).exclude()
    z.add_declaration_code('%s const &pointee_%s(%s const &inst) { return *((%s const *)inst); }' % (full_klass_name, klass_name, z.decl_string[2:], full_klass_name))
    z.add_registration_code('add_property("pointee", bp::make_function(&pointee_%s, bp::return_internal_reference<>()))' % klass_name)
module_builder.module_builder_t.expose_class_Ptr = expose_class_Ptr

def add_doc(self, decl_name, *strings):
    """Adds a few strings to the docstring of declaration f"""
    if len(strings) == 0:
        return
    s = reduce(lambda x, y: x+y, ["\\n    "+x for x in strings])
    self.cc.write('''
_str = "STR"
if DECL.__doc__ is None:
    DECL.__doc__ = _str
else:
    DECL.__doc__ += _str
'''.replace("DECL", decl_name).replace("STR", str(s)))
module_builder.module_builder_t.add_doc = add_doc

def insert_del_interface(self, class_name, del_func_name):
    """Insert an interface to delete the self instance"""
    self.cc.write('''
CLASS_NAME._ownershiplevel = 0

def _CLASS_NAME__del__(self):
    if self._ownershiplevel==1:
        DEL_FUNC_NAME(self)
CLASS_NAME.__del__ = _CLASS_NAME__del__
'''.replace("CLASS_NAME", class_name).replace("DEL_FUNC_NAME", del_func_name))
module_builder.module_builder_t.insert_del_interface = insert_del_interface

def init_class(self, z):
    """Initializes a class z"""
    if not z.partial_decl_string[2:] in common._decls_reg:
        common.register_ti(z.partial_decl_string[2:]) # register the class if not done so
    z.include()
    funs = []
    try:
        funs.extend(z.constructors())
    except RuntimeError:
        pass
    try:
        funs.extend(z.mem_funs())
    except RuntimeError:
        pass
    try:
        funs.extend(z.operators())
    except RuntimeError:
        pass
    common.init_transformers(funs)
    z._funs = funs
    common.add_decl_desc(z)
module_builder.module_builder_t.init_class = init_class


def is_arg_touched(f, arg_name):
    for tr in f._transformer_creators:
        for cell in tr.func_closure:
            if arg_name in cell.cell_contents:
                return True
    return False


def beautify_func_list(self, func_list):
    # fix default values
    func_list = [f for f in func_list if not f.ignore]
    for f in func_list:
        for arg in f.arguments:
            if isinstance(arg.default_value, str):
                repl_list = {
                    'std::basic_string<char, std::char_traits<char>, std::allocator<char> >': 'std::string',
                    'cvPoint': 'cv::Point',
                    'cvTermCriteria': 'cv::TermCriteria',
                    'CV_WHOLE_SEQ': 'cv::Range(0, 0x3fffffff)',
                }
                for z in repl_list:
                    arg.default_value = arg.default_value.replace(z, repl_list[z])

    # function argument int *sizes and int dims
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'sizes' and D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == 'dims' and D.is_integral(arg2.type):
                        f._transformer_creators.append(FT.input_array1d('sizes', 'dims'))
                        break
            if arg.name == '_sizes' and D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == '_ndims' and D.is_integral(arg2.type):
                        f._transformer_creators.append(FT.input_array1d('_sizes', '_ndims'))
                        break
                    if arg2.name == 'dims' and D.is_integral(arg2.type):
                        f._transformer_creators.append(FT.input_array1d('_sizes', 'dims'))
                        break
            if arg.name == '_newsz' and D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == '_newndims' and D.is_integral(arg2.type):
                        f._transformer_creators.append(FT.input_array1d('_newsz', '_newndims'))
                        break

    # function argument std::vector<>
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if "std::vector<" in arg.type.decl_string and 'cv::Mat' not in arg.type.decl_string:
                f._transformer_creators.append(FT.arg_std_vector(arg.name))

    # function argument IplImage *, CvMat *, and CvArr * into cv::Mat
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            for typename in ("::IplImage *", "::IplImage const *",
                "::CvArr *", "::CvArr const *",
                "::CvMat *", "::CvMat const *",
                "::cv::Range const *",):
                if typename in arg.type.decl_string:
                    break
            else:
                continue
            f._transformer_creators.append(FT.input_as_Mat(arg.name))

    # function argument CvPoint2D32f
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.type == D.dummy_type_t("::CvPoint2D32f"):
                f._transformer_creators.append(FT.input_as_Point2f(arg.name))

    # function argument CvRNG * or CvRNG &
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            for typename in ("::CvRNG *", "::CvRNG &"):
                if typename in arg.type.decl_string:
                    break
            else:
                continue
            f._transformer_creators.append(FT.input_asRNG(arg.name))

    # function argument CvFileStorage *
    for f in func_list:
        for arg in f.arguments:
            if not is_arg_touched(f, arg.name) and arg.type == D.dummy_type_t("::CvFileStorage *"):
                f._transformer_creators.append(FT.input_as_FileStorage(arg.name))

    # function argument CvFileNode *
    for f in func_list:
        for arg in f.arguments:
            if not is_arg_touched(f, arg.name) and \
            (arg.type == D.dummy_type_t("::CvFileNode *") or arg.type == D.dummy_type_t("::CvFileNode const *")):
                f._transformer_creators.append(FT.input_as_FileNode(arg.name))

    # function argument CvSparseMat * or CvSparseMat &
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            for typename in ("::CvSparseMat *", "::CvSparseMat &"):
                if arg.type == D.dummy_type_t(typename):
                    break
            else:
                continue
            f._transformer_creators.append(FT.input_asSparseMat(arg.name))

    # function argument const CvPoint2D32f * src and const CvPoint2D32f * dst
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'src' and D.is_pointer(arg.type) and 'CvPoint2D32f' in arg.type.decl_string:
                for arg2 in f.arguments:
                    if arg2.name == 'dst' and D.is_pointer(arg2.type) and 'CvPoint2D32f' in arg2.type.decl_string:
                        f._transformer_creators.append(FT.input_array1d('src'))
                        f._transformer_creators.append(FT.input_array1d('dst'))
                        break

    #  argument 'void *data'
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'data' and D.is_void_pointer(arg.type):
                f._transformer_creators.append(FT.input_string(arg.name))
                mb.add_doc(f.name, "'data' is represented by a string")

    # final step: apply all the function transformations
    for f in func_list:
        if len(f._transformer_creators) > 0:
            f.add_transformation(*f._transformer_creators, **f._transformer_kwds)
            if 'unique_function_name' in f._transformer_kwds:
                f.transformations[0].unique_name = f._transformer_kwds['unique_function_name']
            else:
                s = f.transformations[0].unique_name
                repl_dict = {
                    'operator()': '__call__',
                }
                for t in repl_dict:
                    if t in s:
                        s = s.replace(t, repl_dict[t])
                        f.transformations[0].unique_name = s
                        f.transformations[0].alias = repl_dict[t]
                        break

        common.add_decl_desc(f)

module_builder.module_builder_t.beautify_func_list = beautify_func_list

def finalize_class(self, z):
    """Finalizes a class z"""
    mb.beautify_func_list(z._funs)

    # ignore all non-public members
    for t in z.decls():
        try:
            if t.access_type != 'public' or t.name.startswith('~'):
                t.exclude()
        except:
            pass

    # convert a std::vector<> into something useful
    # try:
        # zz = z.vars()
    # except RuntimeError:
        # zz = []
    # for t in zz:
        # if not t.ignore and 'std::vector' in t.type.decl_string:
            # z.include_files.append("opencv_converters.hpp")
            # t.exclude()
            # z.add_declaration_code('''
# static bp::object get_MEMBER(KLASS const &inst) { return convert_from_T_to_object(inst.MEMBER); }
            # '''.replace('MEMBER', t.name).replace('KLASS', z.decl_string))
            # z.add_registration_code('add_property("MEMBER", &get_MEMBER)'.replace('MEMBER', t.name))

    # if a function returns a pointer and does not have a call policy, create a default one for it
    for f in z._funs:
        if not f.ignore and f.call_policies is None and \
            FT._T.is_ref_or_ptr(f.return_type) and not FT._T.is_ref_or_ptr(FT._T.remove_ref_or_ptr(f.return_type)):
            f.call_policies = CP.return_internal_reference()

module_builder.module_builder_t.finalize_class = finalize_class

def asClass2(self, src_class_Pname, src_class_Cname, dst_class_Pname, dst_class_Cname):
    self.dummy_struct.add_reg_code(\
        'bp::def("asKLASS2", &::normal_cast< CLASS1, CLASS2 >, (bp::arg("inst_KLASS1")));'\
        .replace('KLASS1', src_class_Pname).replace('KLASS2', dst_class_Pname)\
        .replace('CLASS1', src_class_Cname).replace('CLASS2', dst_class_Cname))
module_builder.module_builder_t.asClass2 = asClass2

def dtypecast(self, casting_list):
    for t1 in casting_list:
        z1 = self.class_(t1).alias
        for t2 in casting_list:
            if t1 == t2:
                continue
            z2 = self.class_(t2).alias
            asClass2(self, z1, t1, z2, t2)

module_builder.module_builder_t.dtypecast = dtypecast

def asClass(self, src_class, dst_class):
    asClass2(self, src_class.alias, src_class.partial_decl_string, dst_class.alias, dst_class.partial_decl_string)
    for z in src_class.operators(lambda x: dst_class.name in x.name):
        z.rename('__temp_func')
module_builder.module_builder_t.asClass = asClass



#=============================================================================
# Initialization
#=============================================================================


#Well, don't you want to see what is going on?
# mb.print_declarations() -- too many declarations

# Disable every declarations first
mb.decls().exclude()

# disable some warnings
# mb.decls().disable_warnings(messages.W1027, messages.W1025)

# expose 'this'
mb.classes().expose_this = True

# expose all enumerations
mb.enums().include()

# dummy struct
z = mb.class_("dummy_struct")
z.include_files.append("opencv_converters.hpp")
z.include_files.append("sequence.hpp")
mb.dummy_struct = z
z.include()
z.decls().exclude()
z.class_('dummy_struct2').include()
z.rename("__dummy_struct")
z._reg_code = ""
def add_dummy_reg_code(s):
    mb.dummy_struct._reg_code += "\n        "+s
z.add_reg_code = add_dummy_reg_code

z.add_reg_code("sdcpp::register_sdobject<sdcpp::sequence>();")

# get the list of OpenCV functions
opencv_funs = mb.free_funs() # mb.free_funs(lambda decl: decl.name.startswith('cv'))

# initialize list of transformer creators for each function
common.init_transformers(opencv_funs)

# turn on 'most' of the constants
for z in ('IPL_', 'CV_'):
    try:
        mb.decls(lambda decl: decl.name.startswith(z)).include()
    except RuntimeError:
        pass

# rename some classes
_class_rename = {
    'Point_<float>': 'Point2f',
    'Point3_<int>': 'Point3i',
    'Rect_<int>': 'Rect',
    'Vec<int, 2>': 'Vec2i',
    'Vec<float, 2>': 'Vec2f',
    'Vec<float, 3>': 'Vec3f',
}
for t in _class_rename:
    mb.class_(t).rename(_class_rename[t])



#=============================================================================
# Wrappers for different headers
#=============================================================================

# cxerror.h
print "Generating code for cxerror.h..."
cxerror_h.generate_code(mb, cc, D, FT, CP)

# cxtypes.h
print "Generating code for cxtype.h..."
cxtypes_h.generate_code(mb, cc, D, FT, CP)

# cxcore.h
print "Generating code for cxcore.h..."
cxcore_h.generate_code(mb, cc, D, FT, CP)

# cxcore.hpp
print "Generating code for cxcore.hpp..."
cxcore_hpp.generate_code(mb, cc, D, FT, CP)

# cxoperations.hpp
print "Generating code for cxoperations.hpp..."
cxoperations_hpp.generate_code(mb, cc, D, FT, CP)

# cxflann.h
print "Generating code for cxflann.h..."
cxflann_h.generate_code(mb, cc, D, FT, CP)

# cxmat.hpp
# cxmat_hpp.generate_code(mb, cc, D, FT, CP)

# cvtypes.h
print "Generating code for cvtypes.h..."
cvtypes_h.generate_code(mb, cc, D, FT, CP)

# cv.h
print "Generating code for cv.h..."
cv_h.generate_code(mb, cc, D, FT, CP)

# cv.hpp
print "Generating code for cv.hpp..."
cv_hpp.generate_code(mb, cc, D, FT, CP)

# cvcompat.h
# cvcompat_h.generate_code(mb, cc, D, FT, CP)

# cvaux.h
print "Generating code for cvaux.h..."
cvaux_h.generate_code(mb, cc, D, FT, CP)

# cvaux.hpp
print "Generating code for cvaux.hpp..."
cvaux_hpp.generate_code(mb, cc, D, FT, CP)

# cvvidsurv.hpp
print "Generating code for cvvidsurf.hpp..."
cvvidsurv_hpp.generate_code(mb, cc, D, FT, CP)

# ml.h
print "Generating code for ml.h..."
ml_h.generate_code(mb, cc, D, FT, CP)

# highgui.h
print "Generating code for highgui.h..."
highgui_h.generate_code(mb, cc, D, FT, CP)

# highgui.hpp
print "Generating code for highgui.hpp..."
highgui_hpp.generate_code(mb, cc, D, FT, CP)

# sdopencv
print "Generating code for sdopencv..."
sdopencv.generate_code(mb, cc, D, FT, CP)


#=============================================================================
# Final tasks
#=============================================================================


for z in ('_', 'VARENUM', 'GUARANTEE', 'NLS_FUNCTION', 'POWER_ACTION',
    'PROPSETFLAG', 'PROXY_PHASE', 'PROXY_PHASE', 'SYS', 'XLAT_SIDE',
    'STUB_PHASE',
    ):
    mb.enums(lambda x: x.name.startswith(z)).exclude()
mb.enums(lambda x: x.decl_string.startswith('::std')).exclude()
mb.enums(lambda x: x.decl_string.startswith('::tag')).exclude()

# dummy struct
mb.dummy_struct.add_registration_code('''setattr("v0", 0);
    }
    {
        %s''' % mb.dummy_struct._reg_code)


# rename functions that starts with 'cv'
for z in mb.free_funs():
    if z.alias[:2] == 'cv'and z.alias[2].isupper():
        zz = z.alias[2:]
        if len(zz) > 1 and zz[1].islower():
            zz = zz[0].lower()+zz[1:]
        # print "Old name=", z.alias, " new name=", zz
        z.rename(zz)

mb.beautify_func_list(opencv_funs)

cc.write('''
def __vector__repr__(self):
    n = len(self)
    s = "%s(len=%d, [" % (self.__class__.__name__, n)
    if n==1:
        s += repr(self[0])
    elif n==2:
        s += repr(self[0])+", "+repr(self[1])
    elif n==3:
        s += repr(self[0])+", "+repr(self[1])+", "+repr(self[2])
    elif n==4:
        s += repr(self[0])+", "+repr(self[1])+", "+repr(self[2])+", "+repr(self[3])
    elif n > 4:
        s += repr(self[0])+", "+repr(self[1])+", ..., "+repr(self[n-2])+", "+repr(self[n-1])
    s += "])"
    return s
    
def __vector_tolist(self):
    return [self[i] for i in xrange(len(self))]
    
def __vector_fromlist(cls, obj):
    z = cls()
    for x in obj:
        z.append(x)
    return z
''')    


# expose std::vector, only those with alias starting with 'vector_'
# remember to create operator==() for each element type
for z in mb.classes(lambda x: 'std::vector<' in x.decl_string):
    # check if the class has been registered
    try:
        t = common.get_registered_decl(z.partial_decl_string)
        elem_type = t[1]
        t = common.get_registered_decl(elem_type) # to make sure element type is also registered
    except:
        z.exclude()
        z.set_already_exposed(True)
        continue
    z.include()
    z.add_declaration_code('static inline void resize(%s &inst, size_t num) { inst.resize(num); }' \
        % z.partial_decl_string)
    z.add_registration_code('def("resize", &::resize, ( bp::arg("num") ))')
    cc.write('''
CLASS_NAME.__repr__ = __vector__repr__
CLASS_NAME.tolist = __vector_tolist
CLASS_NAME.fromlist = classmethod(__vector_fromlist)
    '''.replace('CLASS_NAME', z.alias))



#=============================================================================
# Build code
#=============================================================================


#Creating code creator. After this step you should not modify/customize declarations.
mb.build_code_creator( module_name='pyopencvext' )

#Hack os.path.normcase
_old_normcase = OP.normcase
def _new_normcase(s):
    return s
OP.normcase = _new_normcase

#Writing code to file.
mb.split_module('pyopencvext')

common.prepare_decls_registration_code()

#Return old normcase
OP.normcase = _old_normcase

#Write the remaining files
# copyfile('opencv_headers.hpp', 'code/opencv_headers.hpp')

chdir(_cwd)
