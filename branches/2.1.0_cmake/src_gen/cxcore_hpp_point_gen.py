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
sb = sdpypp.SdModuleBuilder('cxcore_hpp_point', number_of_files=3)

sb.cc.write('''
#=============================================================================
# cxcore.hpp -- Point classes
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

Point_dict = 'ifd'

# Vec et al
for i in Vec_dict.keys():
    for suffix in Vec_dict[i]:
        sb.register_ti('cv::Vec', [dtype_dict[suffix], i], 'Vec%d%s' % (i, suffix))

# Point et al
for suffix in Point_dict:
    alias = 'Point2%s' % suffix
    sb.register_ti('cv::Point_', [dtype_dict[suffix]], alias)
    try:
        z = sb.mb.class_(lambda x: x.alias==alias)
    except RuntimeError:
        continue
    sb.init_class(z)
    sb.expose_class_vector(z.partial_decl_string[2:])
    sb.expose_class_vector('std::vector< '+z.partial_decl_string[2:]+' >')
    sb.asClass(z, sb.mb.class_('CvPoint'))
    sb.asClass(z, sb.mb.class_('CvPoint2D32f'))
    try:
        sb.asClass(z, sb.mb.class_('Vec<%s, 2>' % dtype_dict[suffix]))
    except RuntimeError:
        pass
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    sb.add_ndarray_interface(z)
    sb.finalize_class(z)

sb.expose_class_Seq('cv::Point_<int>')

sb.cc.write('''
Point = Point2i
Seq_Point = Seq_Point2i
''')
sb.dtypecast(['::cv::Point_<%s>' % dtype_dict[suffix] for suffix in Point_dict])

# TODO: asPoint = asPoint2i

# Point3 et al
for suffix in Point_dict:
    alias = 'Point3%s' % suffix
    sb.register_ti('cv::Point3_', [dtype_dict[suffix]], alias)
    try:
        z = sb.mb.class_(lambda x: x.alias==alias)
    except RuntimeError:
        continue
    sb.init_class(z)
    sb.expose_class_vector(z.partial_decl_string[2:])
    sb.expose_class_vector('std::vector< '+z.partial_decl_string[2:]+' >')
    sb.asClass(z, sb.mb.class_('CvPoint3D32f'))
    try:
        sb.asClass(z, sb.mb.class_('Vec<%s, 3>' % dtype_dict[suffix]))
    except RuntimeError:
        pass
    sb.add_ndarray_interface(z)
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ", z=" + repr(self.z) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    sb.finalize_class(z)
sb.dtypecast(['::cv::Point3_<%s>' % dtype_dict[suffix] for suffix in Point_dict])

# Point-like operations
for t in ('Point2i', 'Point2f', 'Point2d', 'Point3i', 'Point3f', 'Point3d'):
    c = sb.mb.class_(lambda x: x.alias==t)
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
    


# Mat
z = sb.mb.class_('Mat')
z.include_files.append("opencv_converters.hpp")
sb.init_class(z)
sb.expose_class_vector('cv::Mat')
for t in z.constructors():
    if 'void *' in t.decl_string:
        t.exclude()
for t in ('::IplImage', '::CvMat', 'MatExp'):
    z.decls(lambda x: t in x.decl_string).exclude()
z.mem_funs('setTo').call_policies = CP.return_self()
z.mem_funs('adjustROI').call_policies = CP.return_self()
FT.add_data_interface(z, 'inst.data', 'inst.rows*inst.step', ['ptr', 'data', 'refcount', 'datastart', 'dataend', 'addref', 'release'])
sb.add_ndarray_interface(z)
sb.cc.write('''
def _Mat__repr__(self):
    return "Mat()" if self.empty() else "Mat(rows=" + repr(self.rows) \
        + ", cols=" + repr(self.cols) + ", nchannels=" + repr(self.channels()) \
        + ", depth=" + repr(self.depth()) + "):\\n" + repr(self.ndarray)
Mat.__repr__ = _Mat__repr__
''')
z.add_declaration_code('''
static boost::shared_ptr<cv::Mat> Mat__init1__(bp::object const &arg1)
{
    // None
    if(arg1.ptr() == Py_None) return boost::shared_ptr<cv::Mat>(new cv::Mat());
    
    // cv::Mat const &
    bp::extract<cv::Mat const &> arg1a(arg1);
    if(arg1a.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1a()));
    
    // TODO: here
    PyErr_SetString(PyExc_NotImplementedError, "Unable to construct cv::Mat using the given argument.");
    throw bp::error_already_set(); 
    return boost::shared_ptr<cv::Mat>(new cv::Mat());
}

static boost::shared_ptr<cv::Mat> Mat__init2__(bp::object const &arg1, bp::object const &arg2)
{
    // cv::Size, int
    bp::extract<cv::Size const &> arg1a(arg1);
    bp::extract<int> arg2a(arg2);
    if(arg1a.check() && arg2a.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1a(), arg2a()));
    
    // cv::Mat, cv::Rect
    bp::extract<cv::Mat const &> arg1b(arg1);
    bp::extract<cv::Rect> arg2b(arg2);
    if(arg1b.check() && arg2b.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1b(), arg2b()));
    
    // TODO: here
    PyErr_SetString(PyExc_NotImplementedError, "Unable to construct cv::Mat using the given 2 arguments.");
    throw bp::error_already_set(); 
    return boost::shared_ptr<cv::Mat>(new cv::Mat());
}

static boost::shared_ptr<cv::Mat> Mat__init3__(bp::object const &arg1, bp::object const &arg2, bp::object const &arg3)
{
    // int, int, int
    bp::extract<int> arg1a(arg1);
    bp::extract<int> arg2a(arg2);
    bp::extract<int> arg3a(arg3);
    if(arg1a.check() && arg2a.check() && arg3a.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1a(), arg2a(), arg3a()));
    
    // cv::Size, int, cv::Scalar
    bp::extract<cv::Size const &> arg1b(arg1);
    bp::extract<int> arg2b(arg2);
    bp::extract<cv::Scalar const &> arg3b(arg3);
    if(arg1b.check() && arg2b.check() && arg3b.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1b(), arg2b(), arg3b()));
    
    // cv::Mat, cv::Range, cv::Range
    bp::extract<cv::Mat const &> arg1c(arg1);
    bp::extract<cv::Range const &> arg2c(arg2);
    bp::extract<cv::Range const &> arg3c(arg3);
    if(arg1c.check() && arg2c.check() && arg3c.check()) return boost::shared_ptr<cv::Mat>(new cv::Mat(arg1c(), arg2c(), arg3c()));
    
    // TODO: here
    PyErr_SetString(PyExc_NotImplementedError, "Unable to construct cv::Mat using the given 3 arguments.");
    throw bp::error_already_set(); 
    return boost::shared_ptr<cv::Mat>(new cv::Mat());
}

''')
# workaround to fix a bug in invoking a constructor of Mat
z.add_registration_code('def("__init__", bp::make_constructor(&Mat__init1__, bp::default_call_policies(), ( bp::arg("arg1") )))') 
z.add_registration_code('def("__init__", bp::make_constructor(&Mat__init2__, bp::default_call_policies(), ( bp::arg("arg1"), bp::arg("arg2") )))') 
z.add_registration_code('def("__init__", bp::make_constructor(&Mat__init3__, bp::default_call_policies(), ( bp::arg("arg1"), bp::arg("arg2"), bp::arg("arg3") )))')
# to/from CvMat
z2 = sb.mb.class_('CvMat')
sb.asClass(z, z2)
sb.asClass(z2, z, 'return cv::Mat((CvMat const *)&inst)')
# to/from_list_of_Types
list_dict = {
    'int8': 'char',
    'uint8': 'unsigned char',
    'int16': 'short',
    'uint16': 'unsigned short',
    'int': 'int',
    'float32': 'float',
    'float64': 'double',
    'Vec2b': 'cv::Vec2b',
    'Vec3b': 'cv::Vec3b',
    'Vec4b': 'cv::Vec4b',
    'Vec2s': 'cv::Vec2s',
    'Vec3s': 'cv::Vec3s',
    'Vec4s': 'cv::Vec4s',
    'Vec2w': 'cv::Vec2w',
    'Vec3w': 'cv::Vec3w',
    'Vec4w': 'cv::Vec4w',
    'Vec2i': 'cv::Vec2i',
    'Vec3i': 'cv::Vec3i',
    'Vec4i': 'cv::Vec4i',
    'Vec2f': 'cv::Vec2f',
    'Vec3f': 'cv::Vec3f',
    'Vec4f': 'cv::Vec4f',
    'Vec6f': 'cv::Vec6f',
    'Vec2d': 'cv::Vec2d',
    'Vec3d': 'cv::Vec3d',
    'Vec4d': 'cv::Vec4d',
    'Vec6d': 'cv::Vec6d',
    'Point2i': 'cv::Point2i',
    'Point2f': 'cv::Point2f',
    'Point2d': 'cv::Point2d',
    'Point3i': 'cv::Point3i',
    'Point3f': 'cv::Point3f',
    'Point3d': 'cv::Point3d',
    'Rect': 'cv::Rect',
    # 'Rectf': 'cv::Rectf',
    # 'Rectd': 'cv::Rectd',
    'RotatedRect': 'cv::RotatedRect',
    'Size2i': 'cv::Size2i',
    'Size2f': 'cv::Size2f',
    # 'Size2d': 'cv::Size2d',
    'Scalar': 'cv::Scalar',
    'Range': 'cv::Range',
}
for key in list_dict:
    elem_type = list_dict[key]
    z.add_registration_code('def("to_list_of_%s", &convert_from_Mat_to_seq<%s> )' % (key, elem_type))
    z.add_registration_code('def("from_list_of_%s", &convert_from_seq_to_Mat_object<%s> )' % (key, elem_type))
    z.add_registration_code('staticmethod("from_list_of_%s")' % key)
    try:
        z2 = sb.mb.class_(lambda x: x.alias=='vector_'+key)
        sb.asClass(z, z2, 'return convert_from_Mat_to_vector_of_T<%s>(inst)' % elem_type)
        sb.asClass(z2, z, 'return convert_from_vector_of_T_to_Mat<%s>(inst)' % elem_type)
    except RuntimeError:
        pass
    try:
        z2 = sb.mb.class_(lambda x: x.alias=='vector_vector_'+key)
        sb.asClass(z, z2, 'return convert_from_Mat_to_vector_of_vector_of_T<%s>(inst)' % elem_type)
    except RuntimeError:
        pass
sb.finalize_class(z)

# rewrite the asMat function # TODO: enable this
# sb.cc.write('''
# def reshapeSingleChannel(mat):
    # """Reshapes a Mat object into one that has a single channel.
    
    # The function returns mat itself if it is single-channel.

    # If it is multi-channel, the function invokes mat.reshape() to reshape
    # the object. If the object has a single row, the returning object has
    # rows=mat.cols and cols=mat.channels(). Otherwise, the returning object
    # has rows=mat.rows and cols=mat.cols*mat.channels().    
    # """
    # if mat.channels() != 1:
        # new_mat = mat.reshape(1, mat.cols if mat.rows==1 else mat.rows)
        # if '_depends' in mat.__dict__:
            # new_mat._depends = mat._depends
        # return new_mat
    # return mat
    
# def asMat(obj, force_single_channel=False):
    # """Converts a Python object into a Mat object.
    
    # This general-purpose meta-function uses a simple heuristic method to
    # identify the type of the given Python object in order to convert it into
    # a Mat object. First, it invokes the internal asMat() function of the
    # Python extension to try to convert. If not successful, it assumes the 
    # object is a Python sequence, and converts the object into a std::vector 
    # object whose element type is the type of the first element of the Python 
    # sequence. After that, it converts the std::vector object into a Mat 
    # object by invoking the internal asMat() function again.
    
    # In the case that the above heuristic method does not convert into a Mat
    # object with your intended type and depth, use one of the asvector_...()
    # functions to convert your object into a vector before invoking asMat().
    
    # If 'force_single_channel' is True, the returning Mat is single-channel (by
    # invoking reshapeSingleChannel()). Otherwise, PyOpenCV tries to return a 
    # multi-channel Mat whenever possible.
    # """
    
    # if obj is None:
        # return Mat()
        
    # try:
        # out_mat = eval("_ext.asMat(inst_%s=obj)" % obj.__class__.__name__)
    # except TypeError as e: # Boost.Python.ArgumentError is an unexposed subclass
        # if not e.message.startswith('Python argument types in'):
            # raise e
            
        # z = obj[0]
        # if isinstance(z, int):
            # out_mat = _ext.asMat(inst_vector_int=vector_int.fromlist(obj))
        # elif isinstance(z, float):
            # out_mat = _ext.asMat(inst_vector_float64=vector_float64.fromlist(obj))
        # else:
            # out_mat = eval("_ext.asMat(inst_vector_Type=vector_Type.fromlist(obj))"\
                # .replace("Type", z.__class__.__name__))
    
    # if force_single_channel:
        # return reshapeSingleChannel(out_mat)
    # return out_mat
# asMat.__doc__ = asMat.__doc__ + """
# Docstring of the internal asMat function:

# """ + _ext.asMat.__doc__
# ''')

# Ptr<Mat>
sb.expose_class_Ptr('Mat', 'cv')
sb.expose_class_vector('cv::Ptr< cv::Mat >')    

# Mat_
# Minh-Tri: really bad idea to enable these classes, longer compilation 
# time yet no real gain is observed
# Mat_list = []
# for suffix in "bswifd":
    # for i in xrange(1,5):
        # z = sb.mb.class_(lambda x: x.alias=="Mat"+str(i)+suffix)
        # Mat_list.append("::cv::"+z.name)
        # z.include_files.append("opencv_converters.hpp")
        # z.include()
        # z.constructor(lambda x: len(x.arguments)==4 and '*' in \
            # x.arguments[2].type.partial_decl_string).exclude() # TODO
        # z.mem_funs('adjustROI').call_policies = CP.return_self()
        # for t in ('MatExp', 'vector'):
            # z.decls(lambda x: t in x.decl_string).exclude()
        # for t in ('begin', 'end'):
            # z.decls(t).exclude() # TODO
        # z.operators().exclude() # TODO

# sb.dtypecast(Mat_list)

# MatND
z = sb.mb.class_('MatND')
z.include_files.append("boost/python/make_function.hpp")
z.include_files.append("opencv_converters.hpp")
z.include_files.append("boost/python/str.hpp")
sb.init_class(z)
FT.set_array_item_type_as_size_t(z, 'step')
sb.expose_class_vector('cv::MatND')

z.constructors(lambda x: 'const *' in x.decl_string).exclude()
z.add_declaration_code('''
static boost::shared_ptr<cv::MatND> MatND__init1__(std::vector<int> const &_sizes, int _type)
{
    return boost::shared_ptr<cv::MatND>(new cv::MatND(_sizes.size(), &_sizes[0], _type));
}

static boost::shared_ptr<cv::MatND> MatND__init2__(std::vector<int> const &_sizes, int _type, const cv::Scalar& _s)
{
    return boost::shared_ptr<cv::MatND>(new cv::MatND(_sizes.size(), &_sizes[0], _type, _s));
}

static boost::shared_ptr<cv::MatND> MatND__init3__(const cv::MatND& m, std::vector<cv::Range> const &ranges)
{
    return boost::shared_ptr<cv::MatND>(new cv::MatND(m, &ranges[0]));
}
''')
z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init1__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type") )))')
z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init2__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type"), bp::arg("s") )))')
z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init3__, bp::default_call_policies(), ( bp::arg("m"), bp::arg("ranges") )))')

z.operator('()').exclude() # list of ranges, use std::vector<cv::Range> instead
z.add_declaration_code('''
static cv::MatND MatND__call__(const cv::MatND& inst, std::vector<cv::Range> const &ranges)
{
    return inst(&ranges[0]);
}

''')
z.add_registration_code('def("__call__", bp::make_function(&MatND__call__, bp::default_call_policies(), (bp::arg("ranges"))))')

z.decls(lambda x: 'CvMatND' in x.decl_string).exclude()
sb.asClass(z, sb.mb.class_('Mat'))
z.mem_funs('setTo').call_policies = CP.return_self()
FT.add_data_interface(z, 'inst.data', 'inst.size[0]*inst.step[0]',
    ['ptr', 'data', 'refcount', 'datastart', 'dataend', 'addref', 'release'])
sb.finalize_class(z)
sb.add_ndarray_interface(z)
sb.cc.write('''
def _MatND__repr__(self):
    return "MatND(shape=" + repr(self.ndarray.shape) + ", nchannels=" + repr(self.channels()) \
        + ", depth=" + repr(self.depth()) + "):\\n" + repr(self.ndarray)
MatND.__repr__ = _MatND__repr__
''')



sb.done()
