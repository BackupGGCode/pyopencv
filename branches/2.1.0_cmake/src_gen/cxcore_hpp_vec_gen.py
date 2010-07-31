#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

import function_transformers as FT
from pygccxml import declarations as D
from pyplusplus.module_builder import call_policies as CP
import sdpypp
sb = sdpypp.SdModuleBuilder('cxcore_hpp_vec', number_of_files=9)
sb.load_regs('cxcore_h_reg.sdd')

sb.cc.write('''
#=============================================================================
# cxcore.hpp -- Vec classes
#=============================================================================

''')

#=============================================================================
# Structures
#=============================================================================

dtype_dict = {
    'b': 'unsigned char',
    's': 'short',
    'w': 'unsigned short',
    'i': 'int',
    'f': 'float',
    'd': 'double',
}

Vec_dict = {
    2: 'bswifd',
    3: 'bswifd',
    4: 'bswifd',
    6: 'fd',
}

# Vec et al
for i in Vec_dict.keys():
    for suffix in Vec_dict[i]:
        sb.register_ti('cv::Vec', [dtype_dict[suffix], i], 'Vec%d%s' % (i, suffix))
try:
    zz = sb.mb.classes(lambda z: z.name.startswith('Vec<'))
except RuntimeError:
    zz = []
for z in zz:
    sb.init_class(z)
    sb.expose_class_vector(z.partial_decl_string[2:])
    if z.alias=='Vec2i':
        sb.expose_class_vector('std::vector< '+z.partial_decl_string[2:]+' >')
    z.decls(lambda x: 'CvScalar' in x.partial_decl_string).exclude()
    # sb.asClass(z, sb.mb.class_('CvScalar'))
    z.decl('val').exclude() # use operator[] instead
    sb.add_ndarray_interface(z)
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(" + self.ndarray.__str__() + ")"
KLASS.__repr__ = _KLASS__repr__
    '''.replace('KLASS', z.alias))
    sb.finalize_class(z)
for i in Vec_dict.keys():
    sb.dtypecast(['::cv::Vec<%s, %d>' % (dtype_dict[suffix], i) for suffix in Vec_dict[i]])

# Vec-like operations
for cn in Vec_dict.keys():
    for Tp1 in Vec_dict[cn]:
        c1 = sb.mb.class_(lambda x: x.alias=='Vec%d%s' % (cn, Tp1))
        c1.include_files.append("opencv_converters.hpp")
        a1 = "cv::"+c1.alias

        for Tp2 in Vec_dict[cn]:
            c2 = sb.mb.class_(lambda x: x.alias=='Vec%d%s' % (cn, Tp2))
            a2 = "cv::"+c2.alias
            c1.add_registration_code('def("__iadd__", &__iadd__<CLASS1, CLASS2>, bp::return_self<>() )' \
                .replace("CLASS1", a1).replace("CLASS2", a2))
            c1.add_registration_code('def("__isub__", &__isub__<CLASS1, CLASS2>, bp::return_self<>() )' \
                .replace("CLASS1", a1).replace("CLASS2", a2))

        for t in ('__add__', '__sub__', '__eq__', '__ne__'):
            c1.add_registration_code('def("OPERATOR", &OPERATOR<CLASS1, CLASS1> )' \
                .replace("CLASS1", a1).replace("OPERATOR", t))
        if cn < 6:
            c1.add_registration_code('def("__imul__", &__imul__<CLASS1, TP1>, bp::return_self<>() )' \
                .replace("CLASS1", a1).replace("TP1", dtype_dict[Tp1]))
            c1.add_registration_code('def("__mul__", &__mul__<CLASS1, TP1> )' \
                .replace("CLASS1", a1).replace("TP1", dtype_dict[Tp1]))
            c1.add_registration_code('def("__rmul__", &__rmul__<TP1, CLASS1> )' \
                .replace("CLASS1", a1).replace("TP1", dtype_dict[Tp1]))
        c1.add_registration_code('def("__neg__", &__neg__<CLASS1> )' \
            .replace("CLASS1", a1))
    
    
# Complex et al # TODO: expose operators and conversion to/from np.complex... too
for suffix in Vec_dict[6]:
    sb.register_ti('cv::Complex', [dtype_dict[suffix]], 'Complex%s' % suffix)
try:
    zz = sb.mb.classes(lambda z: z.name.startswith('Complex<'))
except RuntimeError:
    zz = []
for z in zz:
    sb.init_class(z)
    z.decls(lambda t: 'std::complex' in t.decl_string).exclude() # no std::complex please
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(re=" + repr(self.re) + ", im=" + repr(self.im) + ")"
KLASS.__repr__ = _KLASS__repr__
    '''.replace('KLASS', z.alias))
    sb.finalize_class(z)
sb.dtypecast(['::cv::Complex<%s>' % dtype_dict[suffix] for suffix in Vec_dict[6]])

# Scalar et al
sb.register_ti('cv::Scalar_', ['double'], 'Scalar')
z = sb.mb.class_('::cv::Scalar_<double>')
sb.init_class(z)
sb.expose_class_vector(z.partial_decl_string[2:])
z.decls(lambda x: 'CvScalar' in x.partial_decl_string).exclude()
# sb.asClass(z, sb.mb.class_('CvScalar'))
sb.finalize_class(z)
sb.add_ndarray_interface(z)
sb.cc.write('''
def _Scalar__repr__(self):
    return "Scalar(" + self.ndarray.__str__() + ")"
Scalar.__repr__ = _Scalar__repr__
''')

# Scalar operations
c = sb.mb.class_(lambda x: x.alias=='Scalar')
c.include_files.append("opencv_converters.hpp")
a = "cv::"+c.alias
c.add_registration_code('def("__iadd__", &__iadd__<CLASS, CLASS >, bp::return_self<>() )' \
    .replace("CLASS", a))
c.add_registration_code('def("__isub__", &__isub__<CLASS, CLASS >, bp::return_self<>() )' \
    .replace("CLASS", a))
c.add_registration_code('def("__imul__", &__imul__<CLASS, double >, bp::return_self<>() )' \
    .replace("CLASS", a))
for t2 in ('__add__', '__sub__', '__ne__', '__eq__'):
    c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, CLASS> )' \
        .replace("CLASS", a).replace("OPERATOR", t2))
c.add_registration_code('def("__mul__", &__mul__<CLASS, double> )' \
    .replace("CLASS", a))
c.add_registration_code('def("__rmul__", &__rmul__<double, CLASS> )' \
    .replace("CLASS", a))
c.add_registration_code('def("__neg__", &__neg__<CLASS> )' \
    .replace("CLASS", a))
    

# Range
z = sb.mb.class_('Range')
sb.init_class(z)
sb.expose_class_vector(z.partial_decl_string[2:])
z.decls(lambda x: 'CvSlice' in x.partial_decl_string).exclude()
# sb.asClass(z, sb.mb.class_('CvSlice'))
sb.add_ndarray_interface(z)
sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(start=" + repr(self.start) + ", end=" + repr(self.end) + ")"
KLASS.__repr__ = _KLASS__repr__
        
CV_WHOLE_SEQ_END_INDEX = 0x3fffffff
CV_WHOLE_SEQ = Range(0, CV_WHOLE_SEQ_END_INDEX)
CV_WHOLE_ARR  = Range( 0, 0x3fffffff )

'''.replace("KLASS", z.alias))
sb.finalize_class(z)

# Range operations
c = sb.mb.class_(lambda x: x.alias=='Range')
c.include_files.append("opencv_converters.hpp")
a = "cv::"+c.alias
c.add_registration_code('def("__not__", &__not__<CLASS> )' \
    .replace("CLASS", a))
c.add_registration_code('def("__iand__", &__iand__<CLASS, CLASS >, bp::return_self<>() )' \
    .replace("CLASS", a))
for t2 in ('__and__', '__ne__', '__eq__'):
    c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, CLASS> )' \
        .replace("CLASS", a).replace("OPERATOR", t2))
for t2 in ('__add__', '__sub__'):
    c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, int> )' \
        .replace("CLASS", a).replace("OPERATOR", t2))
c.add_registration_code('def("__radd__", &__radd__<int, CLASS> )' \
    .replace("CLASS", a))
    

    
sb.done()
sb.save_regs('cxcore_hpp_vec_reg.sdd')
