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

from os import chdir, getcwd
import os.path as OP
from pygccxml import declarations as D
from pyplusplus import module_builder, messages
import function_transformers as FT
from pyplusplus.module_builder import call_policies as CP
from shutil import copyfile

import cxerror_h
import cxtypes_h
import cxcore_h
import cxcore_hpp
import cxflann_h
import cxmat_hpp
import cvtypes_h
import cv_h
import cv_hpp
import cvcompat_h
import cvaux_h
import cvaux_hpp
import highgui_h
import highgui_hpp
import ml_h

_cwd = getcwd()
chdir(OP.join(OP.split(OP.abspath(__file__))[0], '..', 'src', 'pyopencv'))
_work_dir = getcwd()
print("Working directory changed to: %s" % _work_dir)

#Creating an instance of class that will help you to expose your declarations
mb = module_builder.module_builder_t(
    ["opencv_headers.hpp"],
    gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe",
    working_directory=OP.join(_work_dir, 'pyopencvext'),
    include_paths=[
        r"M:\programming\packages\OpenCV\build\2.0\include",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
    ],
    )

cc = open('__init__.py', 'w')
cc.write('''#!/usr/bin/env python
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
CV_MINOR_VERSION    = 0
CV_SUBMINOR_VERSION = 0
CV_VERSION          = "2.0.0"




''')
mb.cc = cc

# -----------------------------------------------------------------------------------------------
# Subroutines related to writing to the __init__.py file
# -----------------------------------------------------------------------------------------------

def add_ndarray_interface(self, klass):
    klass.include_files.append("ndarray.hpp")
    klass.add_registration_code('def("from_ndarray", &bp::from_ndarray< cv::%s > )' % klass.alias)
    klass.add_registration_code('staticmethod("from_ndarray")'.replace("KLASS", klass.alias))
    self.add_doc(klass.alias+".from_ndarray", "Creates a %s view on an ndarray instance." % klass.alias)
    klass.add_registration_code('add_property("ndarray", &bp::as_ndarray< cv::%s >)' % klass.alias)
    self.add_doc(klass.alias, 
        "Property 'ndarray' provides a numpy.ndarray view on the object.",
        "If you create a reference to 'ndarray', you must keep the object unchanged until your reference is deleted, or Python may crash!",
        "",
        "To create an instance of %s that shares the same data with an ndarray instance, just call:" % klass.alias,
        "    b = %s.from_ndarray(a)" % klass.alias,
        "where 'a' is an ndarray instance. Similarly, to avoid a potential Python crash, you must keep 'a' unchanged until 'b' is deleted.")        
    for t in ('getitem', 'setitem', 'getslice', 'setslice'):
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
    s = reduce(lambda x, y: x+y, ["\\n    [pyopencv] "+x for x in strings])
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
    for fun in funs:
        fun._transformer_creators = []
        fun._transformer_kwds = {}
    z._funs = funs
module_builder.module_builder_t.init_class = init_class


def is_arg_touched(f, arg_name):
    for tr in f._transformer_creators:
        if arg_name in tr.func_closure[1].cell_contents:
            return True
    return False


def beautify_func_list(self, func_list):
    # fix default values
    for f in func_list:
        for arg in f.arguments:
            if isinstance(arg.default_value, str):
                repl_list = {
                    'std::basic_string<char, std::char_traits<char>, std::allocator<char> >': 'std::string',
                    'std::vector<cv::Point_<int>, std::allocator<cv::Point_<int> > >': 'std::vector<cv::Point>',
                    'cvPoint': 'cv::Point',
                    'cvTermCriteria': 'cv::TermCriteria',
                    'CV_WHOLE_SEQ': 'cv::Range(0, 0x3fffffff)',
                    'std::vector<cv::Scalar_<double>, std::allocator<cv::Scalar_<double> > >': 'std::vector<cv::Scalar>',
                    'std::vector<int, std::allocator<int> >': 'std::vector<int>',
                    'std::vector<cv::Vec<int, 4>, std::allocator<cv::Vec<int, 4> > >': 'std::vector<cv::Vec4i>',
                }
                for z in repl_list:
                    arg.default_value = arg.default_value.replace(z, repl_list[z])
                if ", std::allocator<" in arg.default_value: # std::allocator
                    print("func=%s arg.name=%s arg.default_value=%s" % (f.alias, arg.name, arg.default_value))
            
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
            if arg.type.decl_string.startswith("::std::vector<std::vector<"):
                f._transformer_creators.append(FT.input_std_vector_vector(arg.name))
            elif arg.type.decl_string.startswith("::std::vector<"):
                f._transformer_creators.append(FT.input_std_vector(arg.name))

    # function argument IplImage *, CvMat *, CvArr *, and std::vector<> into cv::Mat
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
                if not f.ignore:
                    mb.add_doc(f.name, "'data' is represented by a string")
                    
    # final step: apply all the function transformations
    for f in func_list:
        if len(f._transformer_creators) > 0:
            f.add_transformation(*f._transformer_creators, **f._transformer_kwds)
            
module_builder.module_builder_t.beautify_func_list = beautify_func_list

def finalize_class(self, z):
    """Finalizes a class z"""
    mb.beautify_func_list(z._funs)

    # ignore all non-public members
    for t in z.decls():
        try:
            if t.access_type != 'public':
                t.exclude()
        except:
            pass

    # if a function returns a pointer and does not have a call policy, create a default one for it
    for f in z._funs:
        if not f.ignore and f.call_policies is None and \
            FT._T.is_ref_or_ptr(f.return_type) and not FT._T.is_ref_or_ptr(FT._T.remove_ref_or_ptr(f.return_type)):
            f.call_policies = CP.return_internal_reference()

module_builder.module_builder_t.finalize_class = finalize_class



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

# get the list of OpenCV functions
opencv_funs = mb.free_funs() # mb.free_funs(lambda decl: decl.name.startswith('cv'))

# initialize list of transformer creators for each function
for z in opencv_funs:
    z._transformer_creators = []
    z._transformer_kwds = {}

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

# too many issues when exposing a std::vector as a member variable
# to name a few: missing operators like ==
for z in mb.classes(lambda x: x.name.startswith('vector<')):
    z.exclude() 
    z.set_already_exposed(True)



#=============================================================================
# Wrappers for different headers
#=============================================================================

# cxerror.h
cxerror_h.generate_code(mb, cc, D, FT, CP)

# cxtypes.h
cxtypes_h.generate_code(mb, cc, D, FT, CP)

# cxcore.h
cxcore_h.generate_code(mb, cc, D, FT, CP)

# cxcore.hpp
cxcore_hpp.generate_code(mb, cc, D, FT, CP)

# cxflann.h
cxflann_h.generate_code(mb, cc, D, FT, CP)

# cxmat.hpp
# cxmat_hpp.generate_code(mb, cc, D, FT, CP)

# cvtypes.h
cvtypes_h.generate_code(mb, cc, D, FT, CP)

# cv.h
cv_h.generate_code(mb, cc, D, FT, CP)

# cv.hpp
cv_hpp.generate_code(mb, cc, D, FT, CP)

# cvcompat.h
# cvcompat_h.generate_code(mb, cc, D, FT, CP)

# cvaux.h
cvaux_h.generate_code(mb, cc, D, FT, CP)

# cvaux.hpp
cvaux_hpp.generate_code(mb, cc, D, FT, CP)

# ml.h
ml_h.generate_code(mb, cc, D, FT, CP)

# highgui.h
highgui_h.generate_code(mb, cc, D, FT, CP)

# highgui.hpp
highgui_hpp.generate_code(mb, cc, D, FT, CP)




#=============================================================================
# Rules for free functions and member functions
#=============================================================================


mb.beautify_func_list(opencv_funs)


#=============================================================================
# Final tasks
#=============================================================================


for z in ('_', 'VARENUM', 'GUARANTEE', 'NLS_FUNCTION', 'POWER_ACTION', 
    'PROPSETFLAG', 'PROXY_PHASE', 'PROXY_PHASE', 'SYS', 'XLAT_SIDE',
    ):
    mb.enums(lambda x: x.name.startswith(z)).exclude()
mb.enums(lambda x: x.decl_string.startswith('::std')).exclude()
mb.enums(lambda x: x.decl_string.startswith('::tag')).exclude()

# rename functions that starts with 'cv'
for z in mb.free_funs():
    if z.alias[:2] == 'cv'and z.alias[2].isupper():
        zz = z.alias[2:]
        if len(zz) > 1 and zz[1].islower():
            zz = zz[0].lower()+zz[1:]
        # print "Old name=", z.alias, " new name=", zz
        z.rename(zz)


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

#Return old normcase
OP.normcase = _old_normcase

#Write the remaining files
# copyfile('opencv_headers.hpp', 'code/opencv_headers.hpp')
# copyfile('opencv_extra.hpp', 'code/opencv_extra.hpp')
# copyfile('opencv_extra.cpp', 'code/opencv_extra.cpp')
# copyfile('ndarray.cpp', 'code/ndarray.cpp')

chdir(_cwd)
