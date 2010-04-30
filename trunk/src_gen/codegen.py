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

cc.write('''
def __sd_iter__(self):
    for i in xrange(len(self)):
        yield self[i]

''')

def add_iterator_interface(self, klass_name):
    cc.write('%s.__iter__ = __sd_iter__; ' % klass_name)
module_builder.module_builder_t.add_iterator_interface = add_iterator_interface

def expose_class_Ptr(self, klass_name, ns=None):
    if ns is None:
        full_klass_name = klass_name
    else:
        full_klass_name = '%s::%s' % (ns, klass_name)
    z = self.class_('Ptr<%s>' % full_klass_name)
    common.register_ti('cv::Ptr', [full_klass_name])
    mb.init_class(z)
    # constructor Ptr(_obj) needs to keep a reference of '_obj'
    z.constructors(lambda x: len(x.arguments) > 0).exclude()
    z.operators().exclude()
    z.include_files.append('boost/python/object/life_support.hpp')
    z.add_declaration_code('''
static bp::object from_ELEM_NAME(bp::object const &inst_ELEM_NAME)
{
    bp::extract<ELEM_TYPE *> elem(inst_ELEM_NAME);
    if(!elem.check())
    {
        char s[300];
        sprintf( s, "Argument 'inst_ELEM_NAME' must contain an object of type ELEM_NAME." );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }

    bp::object result = bp::object(CLASS_TYPE(elem()));
    bp::objects::make_nurse_and_patient(result.ptr(), inst_ELEM_NAME.ptr());
    return result;
}

static ELEM_TYPE const &pointee(CLASS_TYPE const &inst) { return *((ELEM_TYPE const *)inst); }
    '''.replace('ELEM_TYPE', full_klass_name).replace('CLASS_TYPE', z.partial_decl_string)\
    .replace('ELEM_NAME', klass_name))
    z.add_registration_code('def("fromELEM_NAME", &::from_ELEM_NAME, (bp::arg("inst_ELEM_NAME")))'\
        .replace('ELEM_NAME', klass_name))
    z.add_registration_code('staticmethod("fromELEM_NAME")'.replace('ELEM_NAME', klass_name))
    z.add_registration_code('add_property("pointee", bp::make_function(&::pointee, bp::return_internal_reference<>()))')
    mb.finalize_class(z)
module_builder.module_builder_t.expose_class_Ptr = expose_class_Ptr

def expose_class_Seq(self, elem_type_pds, pyName=None):
    seq_pds = common.register_ti('cv::Seq', [elem_type_pds], pyName)
    try:
        z = common.find_class(seq_pds)
    except RuntimeError, e:
        print "Cannot determine class with pds='%s'." % seq_pds
        return
    mb.init_class(z)
    z.decls(lambda x: 'CvSeq' in x.partial_decl_string).exclude() # no CvSeq things
    z.constructors(lambda x: len(x.arguments) > 0).exclude()
    z.include_files.append('boost/python/object/life_support.hpp')
    z.add_declaration_code('''
static bp::object from_MemStorage(bp::object const &inst_MemStorage, int headerSize)
{
    bp::extract<cv::MemStorage &> elem(inst_MemStorage);
    if(!elem.check())
    {
        char s[300];
        sprintf( s, "Argument 'inst_MemStorage' must contain an object of type MemStorage." );
        PyErr_SetString(PyExc_TypeError, s);
        throw bp::error_already_set();
    }

    bp::object result = bp::object(CLASS_TYPE(elem(), headerSize));
    bp::objects::make_nurse_and_patient(result.ptr(), inst_MemStorage.ptr());
    return result;
}

static size_t len(CLASS_TYPE const &inst) { return inst.size(); }
    '''.replace('CLASS_TYPE', z.pds))
    z.add_registration_code('def("fromMemStorage", &::from_MemStorage, (bp::arg("inst_MemStorage"), bp::arg("headerSize")=bp::object(sizeof(CvSeq))))')
    z.add_registration_code('staticmethod("fromMemStorage")')
    z.add_registration_code('def("__len__", &::len)')
    for t in ('begin', 'end', 'front', 'back', 'copyTo', 'seq'): # TODO
        z.decls(t).exclude()
    z.mem_funs(lambda x: len(x.arguments)>0 and x.arguments[-1].name=='count').exclude() # TODO
    z.operators(lambda x: 'std::vector' in x.name).exclude() # TODO
    mb.finalize_class(z)
    mb.add_iterator_interface(z.alias)
module_builder.module_builder_t.expose_class_Seq = expose_class_Seq



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
    if not z.pds in common._decls_reg:
        common.register_ti(z.pds) # register the class if not done so
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
    func_list = [f for f in func_list if not f.ignore]

    # fix default values
    # don't remove std::vector default values, old compilers _need_ std::allocator removed
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

    # one-to-one function argument
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            pds = common.unique_pds(arg.type.partial_decl_string)
            if pds=='CvPoint2D32f':
                f._transformer_creators.append(FT.input_as_FixType('CvPoint2D32f', 'cv::Point_<float>', arg.name))
            elif pds=='CvSize':
                f._transformer_creators.append(FT.input_as_FixType('CvSize', 'cv::Size_<int>', arg.name))
            elif pds=='CvSize2D32f':
                f._transformer_creators.append(FT.input_as_FixType('CvSize2D32f', 'cv::Size_<float>', arg.name))
            elif pds=='CvBox2D':
                f._transformer_creators.append(FT.input_as_FixType('CvBox2D', 'cv::RotatedRect', arg.name))
            elif pds=='CvTermCriteria':
                f._transformer_creators.append(FT.input_as_FixType('CvTermCriteria', 'cv::TermCriteria', arg.name))
            elif pds=='CvScalar':
                f._transformer_creators.append(FT.input_as_FixType('CvScalar', 'cv::Scalar_<double>', arg.name))
            elif pds=='CvSlice':
                f._transformer_creators.append(FT.input_as_FixType('CvSlice', 'cv::Range', arg.name))
            elif pds=='CvRect':
                f._transformer_creators.append(FT.input_as_FixType('CvRect', 'cv::Rect_<int>', arg.name))
            elif pds in ['CvRNG *', 'CvRNG &', 'CvRNG cosnt *', 'CvRNG const &']:
                f._transformer_creators.append(FT.input_asRNG(arg.name))
            elif pds in ['CvFileStorage *', 'CvFileStorage const *']:
                f._transformer_creators.append(FT.input_as_FileStorage(arg.name))
            elif pds in ['CvFileNode *', 'CvFileNode const *']:
                f._transformer_creators.append(FT.input_as_FileNode(arg.name))
            elif pds in ['CvMemStorage *', 'CvMemStorage const *']:
                f._transformer_creators.append(FT.input_as_MemStorage(arg.name))
            elif pds in ['CvSparseMat *', 'CvSparseMat &', 'CvSparseMat const *', 'CvSparseMat const &']:
                f._transformer_creators.append(FT.input_asSparseMat(arg.name))
            elif pds in ["IplImage *", "IplImage const *", "CvArr *", "CvArr const *",
                "CvMat *", "CvMat const *", "cv::Range const *"]:
                f._transformer_creators.append(FT.input_as_Mat(arg.name))

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
    FT.beautify_memvars(z)

    # ignore all non-public members
    for t in z.decls():
        try:
            if t.access_type != 'public' or t.name.startswith('~'):
                t.exclude()
        except:
            pass

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

def asClass(self, src_class, dst_class, normal_cast_code=None):
    src_type = src_class.partial_decl_string
    dst_type = dst_class.partial_decl_string
    if normal_cast_code is None:
        for z in src_class.operators(lambda x: dst_class.name in x.name):
            z.rename('__temp_func')
    else:
        self.dummy_struct.add_declaration_code(\
            'template<> inline DstType normal_cast( SrcType const &inst ) { normal_cast_code; }'\
            .replace('normal_cast_code', normal_cast_code)\
            .replace('SrcType', src_type).replace('DstType', dst_type))
    asClass2(self, src_class.alias, src_type, dst_class.alias, dst_type)
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

# make sure size_t is still size_t -- for 64-bit support
z = mb.decl('size_t')
z.type = FT.size_t_t()

# add 'pds' attribute to every class
for z in mb.classes():
    z.pds = common.unique_pds(z.partial_decl_string)

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

def is_vector(cls):
    """Returns whether class 'cls' is a std::vector class."""
    return cls.__name__.startswith('vector_')

def __vector_tolist(self):
    if is_vector(self.elem_type):
        return [self[i].tolist() for i in xrange(len(self))]
    return [self[i] for i in xrange(len(self))]

def __vector_fromlist(cls, obj):
    z = cls()
    if is_vector(cls.elem_type):
        for x in obj:
            z.append(cls.elem_type.fromlist(x))
    else:
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
_z = CLASS_NAME()
_z.resize(1)
CLASS_NAME.elem_type = _z[0].__class__
del(_z)
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
