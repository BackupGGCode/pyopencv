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

import os as _os
import os.path as _op
import sys as _sys
from pygccxml import declarations as _D
import pyplusplus as _pp
from pyplusplus.module_builder import call_policies as _CP
import common as _c
import function_transformers as _FT
import memvar_transformers as _MT


class SdModuleBuilder:
    mb = None
    cc = None
    FT = _FT
    MT = _MT
    D = _D
    CP = _CP
    
    
    def __init__(self, header_file_name, include_paths=[]):
        # module name
        self.module_name = header_file_name.replace(".", "_")
    
        # package directory
        self.pkg_dir = _op.join(_op.split(_op.abspath(__file__))[0], '..', 'src', 'package')
    
        # create an instance of class that will help you to expose your declarations
        self.mb = _pp.module_builder.module_builder_t([header_file_name],
            gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe",
            include_paths=include_paths+[
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
        ''')

        # Well, don't you want to see what is going on?
        # self.mb.print_declarations() -- too many declarations

        # Disable every declarations first
        self.mb.decls().exclude()

        # disable some warnings
        # self.mb.decls().disable_warnings(messages.W1027, messages.W1025)

        # expose 'this'
        self.mb.classes().expose_this = True

        # expose all enumerations
        self.mb.enums().include()

        # add 'pds' attribute to every class
        for z in self.mb.classes():
            z.pds = common.unique_pds(z.partial_decl_string)
            
        # dummy struct # TODO: here
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

        
    def write(self):


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


# rewrite the asndarray function
cc.write('''
def asndarray(obj):
    """Converts a Python object into a numpy.ndarray object.
    
    This function basically invokes:
    
        _PE.asndarray(inst_<type of 'obj'>=obj)
    
    where _PE.asndarray is the internal asndarray() function of the Python
    extension, and the type of the given Python object, 'obj', is determined
    by looking at 'obj.__class__'.
    """
    return eval("_PE.asndarray(inst_%s=obj)" % obj.__class__.__name__)
asndarray.__doc__ = asndarray.__doc__ + """
Docstring of the internal asndarray function:

""" + _PE.asndarray.__doc__
''')
    
for z in ('_', 'VARENUM', 'GUARANTEE', 'NLS_FUNCTION', 'POWER_ACTION',
    'PROPSETFLAG', 'PROXY_PHASE', 'PROXY_PHASE', 'SYS', 'XLAT_SIDE',
    'STUB_PHASE',
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
    
def __vector_create(self, obj):
    """Creates the vector from a Python sequence.
    
    Argument 'obj':
        a Python sequence
    """
    N = len(obj)
    self.resize(N)
    if is_vector(self.elem_type):
        for i in xrange(N):
            self[i] = self.elem_type.fromlist(obj[i])
    else:
        for i in xrange(N):
            self[i] = obj[i]

def __vector_tolist(self):
    if is_vector(self.elem_type):
        return [self[i].tolist() for i in xrange(len(self))]
    return [self[i] for i in xrange(len(self))]

def __vector_fromlist(cls, obj):
    """Creates a vector from a Python sequence.
    
    Argument 'obj':
        a Python sequence
    """
    z = cls()
    z.create(obj)
    return z
    
def __vector__init__(self, obj=None):
    """Initializes the vector.
    
    Argument 'obj':
        If 'obj' is an integer, the vector is initialized as a vector of 
        'obj' elements. If 'obj' is a Python sequence. The vector is
        initialized as an equivalence of 'obj' by invoking self.fromlist().
    """
    self.__old_init__()
    if isinstance(obj, int):
        self.resize(obj)
    elif not obj is None:
        self.create(obj)
    
''')


# expose std::vector, only those with alias starting with 'vector_'
# remember to create operator==() for each element type
for z in mb.classes(lambda x: x.pds.startswith('std::vector<')):
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
CLASS_NAME.__old_init__ = CLASS_NAME.__init__
CLASS_NAME.__init__ = __vector__init__
CLASS_NAME.create = __vector_create
CLASS_NAME.__repr__ = __vector__repr__
CLASS_NAME.tolist = __vector_tolist
CLASS_NAME.fromlist = classmethod(__vector_fromlist)
_z = CLASS_NAME()
_z.resize(1)
CLASS_NAME.elem_type = _z[0].__class__
del(_z)
    '''.replace('CLASS_NAME', z.alias))
    # add conversion between vector and ndarray
    if FT.is_elem_type_fixed_size(elem_type):
        ds = mb.dummy_struct
        ds.include_files.append('ndarray.hpp')
        ds.add_reg_code('bp::def("asndarray", &sdcpp::vector_to_ndarray2< ELEM_TYPE >, (bp::arg("inst_CLASS_NAME")) );' \
            .replace('CLASS_NAME', z.alias).replace('ELEM_TYPE', elem_type))
        ds.add_reg_code('bp::def("asCLASS_NAME", &sdcpp::ndarray_to_vector2< ELEM_TYPE >, (bp::arg("inst_ndarray")) );' \
            .replace('CLASS_NAME', z.alias).replace('ELEM_TYPE', elem_type))

    
# dummy struct
mb.dummy_struct.add_registration_code('''setattr("v0", 0);
    }
    {
        %s''' % mb.dummy_struct._reg_code)


# hack class_t so that py++ uses attribute 'pds' as declaration string
from pyplusplus.decl_wrappers.class_wrapper import class_t
class_t.old_create_decl_string = class_t.create_decl_string
def create_decl_string(self, with_defaults=True):
    if with_defaults and 'pds' in self.__dict__:
        return self.pds
    return self.old_create_decl_string(with_defaults)
class_t.create_decl_string = create_decl_string
    


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
