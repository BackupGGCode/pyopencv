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

import os.path as _op
import os as _os
import sys as _sys
from pygccxml import declarations as _D
import pyplusplus as _pp
from pyplusplus.module_builder import call_policies as _CP
import common as _c
import function_transformers as _FT
import memvar_transformers as _MT


class SdModuleBuilder:
    # FT = _FT
    # MT = _MT
    # _D = _D
    # CP = _CP

    mb = None
    cc = None
    funs = None

    dummy_struct = None
    def add_reg_code(self, s):
        if self.dummy_struct:
            self.dummy_struct._reg_code += "\n        "+s


    def __init__(self, module_name, include_paths=[], number_of_files=1):
        self.module_name = module_name
        self.number_of_files = number_of_files
        _c.current_sb = self

        # package directory
        self.pkg_dir = _op.join(_op.split(_op.abspath(__file__))[0], '..', 'src', 'package')

        # create an instance of class that will help you to expose your declarations
        self.mb = _pp.module_builder.module_builder_t([module_name+"_wrapper.hpp"],
            gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe",
            include_paths=include_paths+[
                _op.join(self.pkg_dir, "extras", "core"),
                _op.join(self.pkg_dir, "extras", "sdopencv"),
                _op.join(self.pkg_dir, module_name+"_ext"),
                r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
                r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
                r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
            ])

        # create a Python file
        self.cc = open(_op.join(self.pkg_dir, self.module_name+'.py'), 'w')
        self.cc.write('''#!/usr/bin/env python
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

import common as _c
import MODULE_NAME_ext as _ext
from MODULE_NAME_ext import *
        '''.replace("MODULE_NAME", module_name))

        # Well, don't you want to see what is going on?
        # self.mb.print_declarations() -- too many declarations

        # Disable every declarations first
        self.mb.decls().exclude()

        # disable some warnings
        # self.mb.decls().disable_warnings(messages.W1027, messages.W1025)

        # expose 'this'
        try:
            self.mb.classes().expose_this = True
        except RuntimeError:
            pass

        # expose all enumerations
        # try:
            # self.mb.enums().include()
        # except RuntimeError:
            # pass

        # except some weird enums
        # for z in ('_', 'VARENUM', 'GUARANTEE', 'NLS_FUNCTION', 'POWER_ACTION',
            # 'PROPSETFLAG', 'PROXY_PHASE', 'PROXY_PHASE', 'SYS', 'XLAT_SIDE',
            # 'STUB_PHASE',
            # ):
            # try:
                # self.mb.enums(lambda x: x.name.startswith(z)).exclude()
            # except RuntimeError:
                # pass
        # for z in ('::std', '::tag'):
            # try:
                # self.mb.enums(lambda x: x.decl_string.startswith(z)).exclude()
            # except RuntimeError:
                # pass

        # add 'pds' attribute to every class
        for z in self.mb.classes():
            z.pds = _c.unique_pds(z.partial_decl_string)

        # dummy struct
        z = self.mb.class_(module_name+"_dummy_struct")
        self.dummy_struct = z
        z.include()
        z.decls().exclude()
        z.class_('dummy_struct2').include()
        z.rename("__"+z.name)
        z._reg_code = ""

        # turn on 'most' of the constants
        for z in ('IPL_', 'CV_'):
            try:
                self.mb.decls(lambda decl: decl.name.startswith(z)).include()
            except RuntimeError:
                pass

        # initialise the list of free functions
        try:
            self.funs = self.mb.free_funs()
        except RuntimeError:
            self.funs = []
        _c.init_transformers(self.funs)
        
        # make sure size_t is still size_t -- for 64-bit support
        z = self.mb.decl('size_t')
        z.type = _FT.size_t_t()

        self.register_basic_data_types()


    def done(self):
        # update registration code
        self.prepare_decls_registration_code()

        # rename functions that starts with 'cv'
        for z in self.funs:
            if z.alias[:2] == 'cv'and z.alias[2].isupper():
                zz = z.alias[2:]
                if len(zz) > 1 and zz[1].islower():
                    zz = zz[0].lower()+zz[1:]
                # print "Old name=", z.alias, " new name=", zz
                z.rename(zz)

        # beautify free functions
        beautify_func_list(self.funs)

        # expose std::vector, only those with alias starting with 'vector_'
        try:
            zz = self.mb.classes(lambda x: x.pds.startswith('std::vector<'))
        except RuntimeError:
            zz = []
        for z in zz:
            # check if the class has been registered
            try:
                t = self.get_registered_decl(z.partial_decl_string)
                if t[0][0]=='_':
                    raise ValueError() # got underscore, exclude it
                elem_type = t[1]
                t = self.get_registered_decl(elem_type) # to make sure element type is also registered
            except:
                z.exclude()
                z.set_already_exposed(True)
                continue
            z.include()
            # remember to create operator==() for each element type
            z.include_files.append("opencv_headers.hpp")
            z.add_declaration_code('static inline void CLASS_NAME_resize(CLASS_TYPE &inst, size_t num) { inst.resize(num); }'.replace("CLASS_NAME", z.alias).replace("CLASS_TYPE", z.partial_decl_string))
            z.add_registration_code('def("resize", &::CLASS_NAME_resize, ( bp::arg("num") ))'.replace("CLASS_NAME", z.alias))
            self.cc.write('''
CLASS_NAME.__old_init__ = CLASS_NAME.__init__
CLASS_NAME.__init__ = _c.__vector__init__
CLASS_NAME.create = _c.__vector_create
CLASS_NAME.__repr__ = _c.__vector__repr__
CLASS_NAME.tolist = _c.__vector_tolist
CLASS_NAME.fromlist = classmethod(_c.__vector_fromlist)
_z = CLASS_NAME()
_z.resize(1)
CLASS_NAME.elem_type = _z[0].__class__
del(_z)
            '''.replace('CLASS_NAME', z.alias))
            # add conversion between vector and ndarray
            if _FT.is_elem_type_fixed_size(elem_type):
                self.dummy_struct.include_files.append('ndarray.hpp')
                self.add_reg_code('bp::def("asndarray", &sdcpp::vector_to_ndarray2< ELEM_TYPE >, (bp::arg("inst_CLASS_NAME")) );' \
                    .replace('CLASS_NAME', z.alias).replace('ELEM_TYPE', elem_type))
                self.add_reg_code('bp::def("asCLASS_NAME", &sdcpp::ndarray_to_vector2< ELEM_TYPE >, (bp::arg("inst_ndarray")) );' \
                    .replace('CLASS_NAME', z.alias).replace('ELEM_TYPE', elem_type))

        # dummy struct
        self.dummy_struct.add_registration_code('''setattr("v0", 0);
    }
    {
        %s''' % self.dummy_struct._reg_code)


        # ----------
        # BUILD CODE
        # ----------

        self.mb.build_code_creator(self.module_name+"_ext")

        # hack os.path.normcase
        _old_normcase = _op.normcase
        def _new_normcase(s):
            return s
        _op.normcase = _new_normcase

        # change current directory
        _cwd = _os.getcwd()
        _os.chdir(self.pkg_dir)

        # write code to file
        if self.number_of_files > 0:
            self.mb.balanced_split_module(self.module_name+"_ext", self.number_of_files)
        else:
            self.mb.split_module(self.module_name+"_ext")

        # return current directory
        _os.chdir(_cwd)

        # return old normcase
        _op.normcase = _old_normcase


    # ==================
    # class registration
    # ==================
    decls_reg = {}
    
    def register_basic_data_types(self):
        # basic data types
        self.register_decl('None', 'void')
        self.register_decl('bool', 'bool')
        self.register_decl('int8', 'char')
        self.register_decl('int8', 'signed char')
        self.register_decl('int8', 'schar')
        self.register_decl('uint8', 'unsigned char')
        self.register_decl('uint8', 'uchar')
        self.register_decl('int16', 'short')
        self.register_decl('int16', 'short int')
        self.register_decl('uint16', 'unsigned short')
        self.register_decl('uint16', 'short unsigned int')
        self.register_decl('uint16', 'ushort')
        self.register_decl('int', 'int')
        self.register_decl('uint', 'unsigned int')
        self.register_decl('long', 'long')
        self.register_decl('ulong', 'unsigned long')
        self.register_decl('float32', 'float')
        self.register_decl('float64', 'double')

    

    def prepare_decls_registration_code(self):
        str = '''#ifndef SD_MODULE_NAME_TEMPLATE_INSTANTIATIONS_HPP
#define SD_MODULE_NAME_TEMPLATE_INSTANTIATIONS_HPP

class MODULE_NAME_dummy_struct {
public:
    struct dummy_struct2 {};
    static const int total_size = 0'''.replace("MODULE_NAME", self.module_name)

        pdss = self.decls_reg.keys()
        pdss.sort()
        for i in xrange(len(pdss)):
            if '<' in pdss[i]: # only instantiate those that need to
                str += '\n        + sizeof(%s)' % pdss[i]

        str += ''';
};

#endif
'''
        filename = self.module_name+'_template_instantiations.hpp'
        if _c.update_file(_op.join(self.pkg_dir, self.module_name+"_ext", filename), str):
            print "Warning: File '%s' has been modified. Re-run the generator." % filename
            _sys.exit(0)

    # get information of a registered class
    def get_registered_decl(self, pds):
        upds = _c.unique_pds(pds)
        try:
            return self.decls_reg[upds]
        except KeyError:
            raise ValueError("Class of pds '%s' has not been registered." % pds)

    def get_registered_decl_name(self, pds):
        upds = _c.unique_pds(pds)
        try:
            return self.decls_reg[upds][0]
        except KeyError:
            return "(C++)"+upds

    def find_classes(self, pds):
        pds = _c.unique_pds(pds)
        return self.mb.classes(lambda x: x.pds==pds)

    def find_class(self, pds):
        pds = _c.unique_pds(pds)
        return self.mb.class_(lambda x: x.pds==pds)

    # pds = partial_decl_string without the preceeding '::'
    def register_decl(self, pyName, pds, cChildName_pds=None, pyEquivName=None):
        upds = _c.unique_pds(pds)
        if upds in self.decls_reg:
            # print "Declaration %s already registered." % pds
            return upds
        if '::' in pds: # assume it is a class
            print "Registration: %s ==> %s..." % (upds, pyName)
            try:
                self.find_class(upds).rename(pyName)
            except RuntimeError:
                # print "Class %s does not exist." % pds
                pass
        self.decls_reg[upds] = (pyName, _c.unique_pds(cChildName_pds), pyEquivName)
        return upds

    # vector template instantiation
    # cName_pds : C name of the class without template element(s)
    # cChildName_pds : C name of the class without template element(s)
    # e.g. if partial_decl_string is '::std::vector<int>' then
    #    cName_pds='std::vector'
    #    cChildName_pds='int'
    def register_vec(self, cName_pds, cChildName_pds, pyName=None, pds=None, pyEquivName=None):
        cupds = _c.unique_pds(cChildName_pds)
        if pyName is None:
            pyName = cName_pds[cName_pds.rfind(':')+1:] + '_' + self.decls_reg[cupds][0]
        if pds is None:
            pds = cName_pds + '< ' + cChildName_pds + ' >'
        return self.register_decl(pyName, pds, cupds, pyEquivName)

    # non-vector template instantiation
    # cName_pds : C name of the class without template element(s)
    # cElemNames_pds : list of the C names of the template element(s)
    # numbers are represented as int, not as str
    # e.g. if partial_decl_string is '::cv::Vec<int, 4>' then
    #    cName_pds='cv::Vec'
    #    cChildName_pds=['int', 4]
    def register_ti(self, cName_pds, cElemNames_pds=[], pyName=None, pds=None):
        if pyName is None:
            pyName = cName_pds[cName_pds.rfind(':')+1:]
            for elem in cElemNames_pds:
                pyName += '_' + (str(elem) if isinstance(elem, int) else self.decls_reg[_c.unique_pds(elem)][0])
        if pds is None:
            pds = cName_pds
            if len(cElemNames_pds)>0:
                pds += '< '
                for elem in cElemNames_pds:
                    pds += (str(elem) if isinstance(elem, int) else elem) + ', '
                pds = pds[:-2] + ' >'
        return self.register_decl(pyName, pds)

    def get_decl_equivname(self, pds):
        z = self.decls_reg[_c.unique_pds(pds)]
        if z[2] is not None:
            return z[2]
        if z[1] is not None:
            return "list of "+get_decl_equivname(z[1])
        return z[0]

        
    # -----------------------------------------------------------------------------------------------
    # Subroutines related to exposing a class or an interface
    # -----------------------------------------------------------------------------------------------
        
    def add_ndarray_interface(self, klass):
        klass.include_files.append("ndarray.hpp")
        # klass.add_registration_code('def("from_ndarray", &sdcpp::from_ndarray< %s >, (bp::arg("inst_ndarray")) )' % klass.pds)
        try:
            self.mb.included_ndarray += 1
        except AttributeError:
            self.mb.included_ndarray = 1
            self.mb.add_declaration_code('''
#include "ndarray.hpp"
''')
        self.mb.add_registration_code('bp::def("as%s", &sdcpp::from_ndarray< %s >, (bp::arg("inst_ndarray")) );' % (klass.alias, klass.pds))
        # klass.add_registration_code('staticmethod("from_ndarray")')
        # self.add_doc(klass.alias+".from_ndarray", "Creates a %s view on an ndarray instance." % klass.alias)
        klass.add_registration_code('add_property("ndarray", &sdcpp::as_ndarray< %s >)' % klass.pds)
        self.mb.add_registration_code('bp::def("asndarray", &sdcpp::as_ndarray< %s >, (bp::arg("inst_%s")) );' % (klass.pds, klass.alias))
        self.add_doc(klass.alias,
            "Property 'ndarray' provides a numpy.ndarray view on the object.",
            "If you create a reference to 'ndarray', you must keep the object unchanged until your reference is deleted, or Python may crash!",
            "Alternatively, you could create a reference to 'ndarray' by using 'asndarray(obj)', where 'obj' is an instance of this class.",
            "",
            "To create an instance of KLASS that shares the same data with an ndarray instance, use: 'asKLASS(a),".replace("KLASS", klass.alias),
            # "    '%s.from_ndarray(a)' or 'as%s(a)" % (klass.alias, klass.alias),
            "where 'a' is an ndarray instance. Similarly, to avoid a potential Python crash, you must keep the current instance unchanged until the reference is deleted.")
        for t in ('getitem', 'setitem', 'getslice', 'setslice', 'iter'):
            self.cc.write('''
def _KLASS__FUNC__(self, *args, **kwds):
    return self.ndarray.__FUNC__(*args, **kwds)
KLASS.__FUNC__ = _KLASS__FUNC__
            '''.replace('KLASS', klass.alias).replace('FUNC', t))

    def add_iterator_interface(self, klass_name):
        self.cc.write('''
%s.__iter__ = _c.__sd_iter__;
        ''' % klass_name)

    def expose_class_Ptr(self, klass_name, ns=None):
        if ns is None:
            full_klass_name = klass_name
        else:
            full_klass_name = '%s::%s' % (ns, klass_name)
        self.register_ti('cv::Ptr', [full_klass_name])
        try:
            z = self.mb.class_('Ptr<%s>' % full_klass_name)
        except RuntimeError:
            print "Error: Cannot expose class 'Ptr<%s>' because it does not exist." % klass_name
            return
        self.init_class(z)
        # constructor Ptr(_obj) needs to keep a reference of '_obj'
        z.constructors(lambda x: len(x.arguments) > 0).exclude()
        z.operators().exclude()
        z.include_files.append('boost/python/object/life_support.hpp')
        z.add_declaration_code('''
static bp::object CLASS_NAME_from_ELEM_NAME(bp::object const &inst_ELEM_NAME)
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
    result.attr("_depends") = inst_ELEM_NAME;
    return result;
}

static ELEM_TYPE const &CLASS_NAME_pointee(CLASS_TYPE const &inst) { return *((ELEM_TYPE const *)inst); }
        '''.replace('ELEM_TYPE', full_klass_name).replace('CLASS_TYPE', z.partial_decl_string)\
        .replace('CLASS_NAME', z.alias).replace('ELEM_NAME', klass_name))
        z.add_registration_code('def("fromELEM_NAME", &::CLASS_NAME_from_ELEM_NAME, (bp::arg("inst_ELEM_NAME")))'\
            .replace('ELEM_NAME', klass_name).replace('CLASS_NAME', z.alias))
        z.add_registration_code('staticmethod("fromELEM_NAME")'.replace('ELEM_NAME', klass_name))
        z.add_registration_code('add_property("pointee", bp::make_function(&::CLASS_NAME_pointee, bp::return_internal_reference<>()))'.replace('CLASS_NAME', z.alias))
        self.finalize_class(z)

    def expose_class_Seq(self, elem_type_pds, pyName=None):
        seq_pds = self.register_ti('cv::Seq', [elem_type_pds], pyName)
        try:
            z = self.find_class(seq_pds)
        except RuntimeError, e:
            print "Cannot determine class with pds='%s'." % seq_pds
            return
        self.init_class(z)
        # Main problem is that fake constructors don't work with with_custodian_and_ward.
        # I'm using an old trick to circumvent the problem.
        self.cc.write('''
CLASS_NAME.__old_init__ = CLASS_NAME.__init__
def _CLASS_NAME__init__(self, *args, **kwds):
    CLASS_NAME.__old_init__(self, *args, **kwds)
    if args:
        self.depends = [args[0]]
    elif kwds:
        self.depends = [kwds.values()[0]]
    else:
        self.depends = []
_CLASS_NAME__init__.__doc__ = CLASS_NAME.__old_init__.__doc__    
CLASS_NAME.__init__ = _CLASS_NAME__init__
        '''.replace("CLASS_NAME", z.alias))
        z.add_declaration_code('''
static size_t CLASS_NAME_len(CLASS_TYPE const &inst) { return inst.size(); }
        '''.replace("CLASS_NAME", z.alias).replace('CLASS_TYPE', z.pds))
        z.add_registration_code('def("__len__", &::CLASS_NAME_len)'.replace("CLASS_NAME", z.alias))
        for t in ('begin', 'end', 'front', 'back'): # TODO
            z.decls(t).exclude()
        # OpenCV has a bug at function insert()
        # for t in z.mem_funs(lambda x: len(x.arguments)>0 and x.arguments[-1].name=='count'):
            # t._transformer_creators.append(FT.input_array1d(t.arguments[-2].name, 'count'))
            # t._transformer_kwds['alias'] = t.alias
        z.mem_funs(lambda x: len(x.arguments)>0 and x.arguments[-1].name=='count').exclude()
        self.asClass(z, self.find_class('std::vector< %s >' % elem_type_pds))
        self.finalize_class(z)
        self.add_iterator_interface(z.alias)

        
        
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

    def insert_del_interface(self, class_name, del_func_name):
        """Insert an interface to delete the self instance"""
        self.cc.write('''
CLASS_NAME._ownershiplevel = 0

def _CLASS_NAME__del__(self):
    if self._ownershiplevel==1:
        DEL_FUNC_NAME(self)
CLASS_NAME.__del__ = _CLASS_NAME__del__
'''.replace("CLASS_NAME", class_name).replace("DEL_FUNC_NAME", del_func_name))

    def init_class(self, z):
        """Initializes a class z"""
        if not z.pds in self.decls_reg:
            self.register_ti(z.pds) # register the class if not done so
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
        _c.init_transformers(funs)
        z._funs = funs
        _c.add_decl_desc(z)

    def finalize_class(self, z):
        """Finalizes a class z"""
        beautify_func_list(z._funs)
        _FT.beautify_memvars(z)

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
                _FT._T.is_ref_or_ptr(f.return_type) and not _FT._T.is_ref_or_ptr(_FT._T.remove_ref_or_ptr(f.return_type)):
                f.call_policies = _CP.return_internal_reference()

    def asClass2(self, src_class_Pname, src_class_Cname, dst_class_Pname, dst_class_Cname):
        self.dummy_struct.include_files.append("opencv_converters.hpp")
        self.add_reg_code(\
            'bp::def("asKLASS2", &::normal_cast< CLASS1, CLASS2 >, (bp::arg("inst_KLASS1")));'\
            .replace('KLASS1', src_class_Pname).replace('KLASS2', dst_class_Pname)\
            .replace('CLASS1', src_class_Cname).replace('CLASS2', dst_class_Cname))
                
    def dtypecast(self, casting_list):
        for t1 in casting_list:
            try:
                z1 = self.mb.class_(t1).alias
            except RuntimeError:
                continue
            for t2 in casting_list:
                if t1 == t2:
                    continue
                try:
                    z2 = self.mb.class_(t2).alias
                except RuntimeError:
                    continue
                self.asClass2(z1, t1, z2, t2)

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
        self.asClass2(src_class.alias, src_type, dst_class.alias, dst_type)
        
        

def is_arg_touched(f, arg_name):
    for tr in f._transformer_creators:
        for cell in tr.func_closure:
            if arg_name in cell.cell_contents:
                return True
    return False
        
def sort_transformers(f):
    # list of function arguments
    f_args = [x.name for x in f.arguments]

    arg_idx = {}
    for idx in xrange(len(f._transformer_creators)):
        t = f._transformer_creators[idx]
        # get the argument index
        t_args = t.func_closure[0].cell_contents
        if not isinstance(t_args, tuple):
            t_args = t.func_closure[1].cell_contents        
        for ta in t_args:
            if ta in f_args:
                arg_idx[f_args.index(ta)] = idx
                break
        else:
            arg_idx[1000+idx] = idx
    
    # rewrite
    ids = arg_idx.keys()
    ids.sort()
    f._transformer_creators = [f._transformer_creators[arg_idx[id]] for id in ids]

def beautify_func_list(func_list):
    func_list = [f for f in func_list if not f.ignore]

    # fix default values
    # don't remove std::vector default values, old compilers _need_ std::allocator removed
    for f in func_list:
        for arg in f.arguments:
            if isinstance(arg.default_value, str):
                repl_list = {
                    'std::basic_string<char, std::char_traits<char>, std::allocator<char> >': 'std::string',
                    'std::vector<cv::Point_<int>, std::allocator<cv::Point_<int> > >': 'std::vector<cv::Point>',
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
            pds = _c.unique_pds(arg.type.partial_decl_string)
            if pds in _c.c2cpp:
                f._transformer_creators.append(_FT.input_as_FixType(pds, _c.c2cpp[pds], arg.name))
            elif pds in ['CvRNG *', 'CvRNG &', 'CvRNG cosnt *', 'CvRNG const &']:
                f._transformer_creators.append(_FT.input_asRNG(arg.name))
            elif pds in ['CvFileStorage *', 'CvFileStorage const *']:
                f._transformer_creators.append(_FT.input_as_FileStorage(arg.name))
            elif pds in ['CvFileNode *', 'CvFileNode const *']:
                f._transformer_creators.append(_FT.input_as_FileNode(arg.name))
            elif pds in ['CvMemStorage *', 'CvMemStorage const *']:
                f._transformer_creators.append(_FT.input_as_MemStorage(arg.name))
            elif pds in ['CvSparseMat *', 'CvSparseMat &', 'CvSparseMat const *', 'CvSparseMat const &']:
                f._transformer_creators.append(_FT.input_asSparseMat(arg.name))
            elif pds in ["IplImage *", "IplImage const *", "CvArr *", "CvArr const *",
                "CvMat *", "CvMat const *", "cv::Range const *"]:
                f._transformer_creators.append(_FT.input_as_Mat(arg.name))

    # function argument int *sizes and int dims
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'sizes' and _D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == 'dims' and _D.is_integral(arg2.type):
                        f._transformer_creators.append(_FT.input_array1d('sizes', 'dims'))
                        break
            if arg.name == '_sizes' and _D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == '_ndims' and _D.is_integral(arg2.type):
                        f._transformer_creators.append(_FT.input_array1d('_sizes', '_ndims'))
                        break
                    if arg2.name == 'dims' and _D.is_integral(arg2.type):
                        f._transformer_creators.append(_FT.input_array1d('_sizes', 'dims'))
                        break
            if arg.name == '_newsz' and _D.is_pointer(arg.type):
                for arg2 in f.arguments:
                    if arg2.name == '_newndims' and _D.is_integral(arg2.type):
                        f._transformer_creators.append(_FT.input_array1d('_newsz', '_newndims'))
                        break

    # function argument const CvPoint2D32f * src and const CvPoint2D32f * dst
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'src' and _D.is_pointer(arg.type) and 'CvPoint2D32f' in arg.type.decl_string:
                for arg2 in f.arguments:
                    if arg2.name == 'dst' and _D.is_pointer(arg2.type) and 'CvPoint2D32f' in arg2.type.decl_string:
                        f._transformer_creators.append(_FT.input_array1d('src'))
                        f._transformer_creators.append(_FT.input_array1d('dst'))
                        break

    #  argument 'void *data'
    for f in func_list:
        for arg in f.arguments:
            if is_arg_touched(f, arg.name):
                continue
            if arg.name == 'data' and _D.is_void_pointer(arg.type):
                f._transformer_creators.append(_FT.input_string(arg.name))
                self.add_doc(f.name, "'data' is represented by a string")

    # final step: apply all the function transformations
    for f in func_list:
        if len(f._transformer_creators) > 0:
            sort_transformers(f)

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

        _c.add_decl_desc(f)

