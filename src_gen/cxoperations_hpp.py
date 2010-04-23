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

import common

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# cxoperations.hpp
#=============================================================================


    ''')

    dtype_dict = {
        'b': 'unsigned char',
        's': 'short',
        'w': 'unsigned short',
        'i': 'int',
        'f': 'float',
        'd': 'double',
    }
    
    # Vec-like
    Vec_str_from_cn = {
        2: 'bswifd',
        3: 'bswifd',
        4: 'bswifd',
        6: 'fd'
    }
    
    for cn in Vec_str_from_cn.keys():
        for Tp1 in Vec_str_from_cn[cn]:
            c1 = mb.class_(lambda x: x.alias=='Vec%d%s' % (cn, Tp1))
            c1.include_files.append("opencv_converters.hpp")
            a1 = "cv::"+c1.alias

            for Tp2 in Vec_str_from_cn[cn]:
                c2 = mb.class_(lambda x: x.alias=='Vec%d%s' % (cn, Tp2))
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
        
    # Complex-like # TODO: expose operators and conversion to/from np.complex... too
    
    # Point-like
    for t in ('Point2i', 'Point2f', 'Point2d', 'Point3i', 'Point3f', 'Point3d'):
        c = mb.class_(lambda x: x.alias==t)
        c.include_files.append("opencv_converters.hpp")
        a = "cv::"+c.alias
        c.add_registration_code('def("__iadd__", &__iadd__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__isub__", &__isub__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__imul__", &__imul__<CLASS, double>, bp::return_self<>() )' \
            .replace("CLASS", a))
        for t2 in ('__add__', '__sub__', '__eq__'): # '__ne__'
            c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, CLASS> )' \
                .replace("CLASS", a).replace("OPERATOR", t2))
        c.add_registration_code('def("__neg__", &__neg__<CLASS> )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__mul__", &__mul__<CLASS, double> )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__rmul__", &__rmul__<double, CLASS> )' \
            .replace("CLASS", a))
        
    # Size-like
    for t in ('Size2i', 'Size2f'):
        c = mb.class_(lambda x: x.alias==t)
        c.include_files.append("opencv_converters.hpp")
        a = "cv::"+c.alias
        c.add_registration_code('def("__iadd__", &__iadd__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__isub__", &__isub__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        for t2 in ('__add__', '__sub__', '__eq__', '__ne__'):
            c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, CLASS> )' \
                .replace("CLASS", a).replace("OPERATOR", t2))
        c.add_registration_code('def("__mul__", &__mul__<CLASS, TP1> )' \
            .replace("CLASS", a).replace("TP1", c.var('width').type.partial_decl_string))
        
    # Rect-like
    for dtype in ('int',):
        c = mb.class_('::cv::Rect_<%s>' % dtype)
        c.include_files.append("opencv_converters.hpp")
        a = "cv::"+c.alias
        c.add_registration_code('def("__iadd__", &__iadd__<CLASS, cv::Point_<DTYPE> >, bp::return_self<>() )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__iadd__", &__iadd__<CLASS, cv::Size_<DTYPE> >, bp::return_self<>() )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__isub__", &__isub__<CLASS, cv::Point_<DTYPE> >, bp::return_self<>() )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__isub__", &__isub__<CLASS, cv::Size_<DTYPE> >, bp::return_self<>() )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__iand__", &__iand__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        c.add_registration_code('def("__ior__", &__ior__<CLASS, CLASS>, bp::return_self<>() )' \
            .replace("CLASS", a))
        for t2 in ('__and__', '__or__', '__eq__'):
            c.add_registration_code('def("OPERATOR", &OPERATOR<CLASS, CLASS> )' \
                .replace("CLASS", a).replace("OPERATOR", t2))
        c.add_registration_code('def("__add__", &__add__<CLASS, cv::Point_<DTYPE> > )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__sub__", &__sub__<CLASS, cv::Point_<DTYPE> > )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        c.add_registration_code('def("__add__", &__add__<CLASS, cv::Size_<DTYPE> > )' \
            .replace("CLASS", a).replace("DTYPE", dtype))
        
    # Scalar
    c = mb.class_(lambda x: x.alias=='Scalar')
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
    c = mb.class_(lambda x: x.alias=='Range')
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
        
    z = None
    
    # VectorBase
    z = mb.class_('VectorBase')
    mb.init_class(z)
    mb.finalize_class(z)
        
    # Vector
    Vector_list = [
        'int', 'cv::Vec< float, 2 >',
        'cv::SdVector< int >', 'cv::SdVector< cv::Vec< float, 2 > >',
    ]
    for elem in Vector_list:
        common.register_vec('cv::SdVector', elem, 'Vector_' + common._decls_reg[elem][0])
        try:
            cont = False
            z = mb.class_(lambda x: x.partial_decl_string=='::cv::SdVector< %s >' % elem)
        except RuntimeError:
            cont = True
        if cont:
            continue
        mb.init_class(z)
        # remove typedefs and vars and mem_funs
        for t in ('value_type', 'iterator', 'const_iterator', 'reference', 
            'const_reference', 'Hdr', 'begin', 'end', 'set', 'setitem'):
            z.decls(t).exclude()
        for t in ('push_back', 'pop_back'):
            z.mem_fun(t).call_policies = CP.return_self()
        for t in ('back', 'front'):
            z.mem_funs(t).exclude()
                
        z.operator(lambda x: x.name=='operator[]' and 'const' in x.partial_decl_string).exclude()
        z.decls(lambda x: 'std::vector' in x.partial_decl_string).exclude()
        # remove share constructors
        z.constructors(lambda x: len(x.arguments)>1 and \
            x.arguments[len(x.arguments)-1].name=='_copyData').exclude()
        # TODO: constructor from Vec to Vector
        mb.asClass(z, mb.class_('Mat'))
        
        # make an iterator -- TODO: use an iterator instead of list
        z.add_declaration_code('''
static bp::object iter(%s const &inst)
{
    bp::list l;
    for(int i = 0; i < inst.size(); ++i) l.append(bp::object(inst[i]));
    return l.attr("__iter__")();
}        
        ''' % z.partial_decl_string)
        z.add_registration_code('def("__iter__", &::iter)')
        # TODO: make ndarray
        z.add_registration_code('def("__len__", &%s::size)' % z.partial_decl_string)
        z.add_registration_code('def("__setitem__", &%s::setitem)' % z.partial_decl_string)
        mb.finalize_class(z)
        