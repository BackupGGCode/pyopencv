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
# cxcore.hpp
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
    
    common.register_ti('cv::Mat')
    common.register_vec('std::vector', 'char', 'vector_int8', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned char', 'vector_uint8', pyEquivName='Mat')
    common.register_vec('std::vector', 'short', 'vector_int16', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned short', 'vector_uint16', pyEquivName='Mat')
    common.register_vec('std::vector', 'int', 'vector_int', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned int', 'vector_uint', pyEquivName='Mat')
    common.register_vec('std::vector', 'long', 'vector_long', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned long', 'vector_ulong', pyEquivName='Mat')
    common.register_vec('std::vector', 'long long', 'vector_int64', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned long long', 'vector_uint64', pyEquivName='Mat')
    common.register_vec('std::vector', 'float', 'vector_float32', pyEquivName='Mat')
    common.register_vec('std::vector', 'double', 'vector_float64', pyEquivName='Mat')
    common.register_vec('std::vector', 'unsigned char *', 'vector_string')
    common.register_vec('std::vector', 'std::vector< int >', 'vector_vector_int')
    common.register_vec('std::vector', 'std::vector< float >', 'vector_vector_float32')
    
    # Vec et al
    for i in Vec_dict.keys():
        for suffix in Vec_dict[i]:
            common.register_ti('cv::Vec', [dtype_dict[suffix], i], 'Vec%d%s' % (i, suffix))
    zz = mb.classes(lambda z: z.name.startswith('Vec<'))
    for z in zz:
        mb.init_class(z)
        common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
        if z.alias=='Vec2i':
            common.register_vec('std::vector', 'std::vector< '+z.partial_decl_string[2:]+' >')
        z.decls(lambda x: 'CvScalar' in x.partial_decl_string).exclude()
        # mb.asClass(z, mb.class_('CvScalar'))
        z.decl('val').exclude() # use operator[] instead
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(" + self.ndarray.__str__() + ")"
KLASS.__repr__ = _KLASS__repr__
        '''.replace('KLASS', z.alias))
        mb.finalize_class(z)
    for i in Vec_dict.keys():
        mb.dtypecast(['::cv::Vec<%s, %d>' % (dtype_dict[suffix], i) for suffix in Vec_dict[i]])
        
    # Complex et al
    for suffix in Vec_dict[6]:
        common.register_ti('cv::Complex', [dtype_dict[suffix]], 'Complex%s' % suffix)
    for z in mb.classes(lambda z: z.name.startswith('Complex<')):
        mb.init_class(z)
        z.decls(lambda t: 'std::complex' in t.decl_string).exclude() # no std::complex please
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(re=" + repr(self.re) + ", im=" + repr(self.im) + ")"
KLASS.__repr__ = _KLASS__repr__
        '''.replace('KLASS', z.alias))
        mb.finalize_class(z)
    mb.dtypecast(['::cv::Complex<%s>' % dtype_dict[suffix] for suffix in Vec_dict[6]])
    
    # Point et al
    for suffix in Point_dict:
        alias = 'Point2%s' % suffix
        common.register_ti('cv::Point_', [dtype_dict[suffix]], alias)
        z = mb.class_(lambda x: x.alias==alias)
        mb.init_class(z)
        common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
        common.register_vec('std::vector', 'std::vector< '+z.partial_decl_string[2:]+' >')
        mb.asClass(z, mb.class_('CvPoint'))
        mb.asClass(z, mb.class_('CvPoint2D32f'))
        mb.asClass(z, mb.class_('Vec<%s, 2>' % dtype_dict[suffix])) 
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.add_ndarray_interface(z)
        mb.finalize_class(z)
    
    cc.write('''
Point = Point2i
asPoint = asPoint2i
    ''')
    mb.dtypecast(['::cv::Point_<%s>' % dtype_dict[suffix] for suffix in Point_dict])
    
    # Point3 et al
    for suffix in Point_dict:
        alias = 'Point3%s' % suffix
        common.register_ti('cv::Point3_', [dtype_dict[suffix]], alias)
        z = mb.class_(lambda x: x.alias==alias)
        mb.init_class(z)
        common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
        common.register_vec('std::vector', 'std::vector< '+z.partial_decl_string[2:]+' >')
        mb.asClass(z, mb.class_('CvPoint3D32f'))
        mb.asClass(z, mb.class_('Vec<%s, 3>' % dtype_dict[suffix]))
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ", z=" + repr(self.z) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.finalize_class(z)
    mb.dtypecast(['::cv::Point3_<%s>' % dtype_dict[suffix] for suffix in Point_dict])
    
    # Size et al
    Size_dict = 'if'
    for suffix in Size_dict:
        alias = 'Size2%s' % suffix
        common.register_ti('cv::Size_', [dtype_dict[suffix]], alias)
        z = mb.class_(lambda x: x.alias==alias)
        mb.init_class(z)
        common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
        z.decls(lambda x: 'CvSize' in x.partial_decl_string).exclude()
        # mb.asClass(z, mb.class_('CvSize'))
        # mb.asClass(z, mb.class_('CvSize2D32f'))
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.finalize_class(z)
    mb.dtypecast(['::cv::Size_<%s>' % dtype_dict[suffix] for suffix in Size_dict])
        
    cc.write('''
Size = Size2i
    ''')
    
    # Rect
    common.register_ti('cv::Rect_', ['int'], 'Rect')
    z = mb.class_(lambda x: x.alias=='Rect')
    mb.init_class(z)
    common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
    z.decls(lambda x: 'CvRect' in x.partial_decl_string).exclude()
    # mb.asClass(z, mb.class_('CvRect'))
    mb.add_ndarray_interface(z)
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + \\
        ", width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    mb.finalize_class(z)
    # mb.dtypecast(['::cv::Rect_<%s>' % dtype_dict[suffix] for suffix in Point_dict])
    
    # RotatedRect
    z = mb.class_('RotatedRect')
    mb.init_class(z)
    common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
    mb.decls(lambda x: 'CvBox2D' in x.partial_decl_string).exclude()
    # mb.asClass(z, mb.class_('CvBox2D'))
    mb.add_ndarray_interface(z)
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(center=" + repr(self.center) + ", size=" + repr(self.size) + \\
        ", angle=" + repr(self.angle) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    mb.finalize_class(z)
    
    # Scalar et al
    common.register_ti('cv::Scalar_', ['double'], 'Scalar')
    z = mb.class_('::cv::Scalar_<double>')
    mb.init_class(z)
    common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
    z.decls(lambda x: 'CvScalar' in x.partial_decl_string).exclude()
    # mb.asClass(z, mb.class_('CvScalar'))
    mb.finalize_class(z)
    mb.add_ndarray_interface(z)
    cc.write('''
def _Scalar__repr__(self):
    return "Scalar(" + self.ndarray.__str__() + ")"
Scalar.__repr__ = _Scalar__repr__
    ''')
    
    # Range
    z = mb.class_('Range')
    mb.init_class(z)
    common.register_vec('std::vector', z.partial_decl_string[2:], pyEquivName='Mat')
    z.decls(lambda x: 'CvSlice' in x.partial_decl_string).exclude()
    # mb.asClass(z, mb.class_('CvSlice'))
    mb.add_ndarray_interface(z)
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(start=" + repr(self.start) + ", end=" + repr(self.end) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    mb.finalize_class(z)
    
    # Ptr -- already exposed by mb.expose_class_Ptr
    
    # Mat
    z = mb.class_('Mat')
    z.include_files.append("opencv_converters.hpp")
    mb.init_class(z)
    common.register_vec('std::vector', 'cv::Mat')
    for t in z.constructors():
        if 'void *' in t.decl_string:
            t.exclude()
    for t in ('::IplImage', '::CvMat', 'MatExp'):
        z.decls(lambda x: t in x.decl_string).exclude()
    z.mem_funs('setTo').call_policies = CP.return_self()
    z.mem_funs('adjustROI').call_policies = CP.return_self()
    for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        z.decls(t).exclude()
    z.add_declaration_code('''
static bp::object get_data(cv::Mat const &inst)
{
    return bp::object(bp::handle<>(bp::borrowed(PyBuffer_FromReadWriteMemory(
        (void*)inst.data, inst.rows*inst.step))));
}

    ''')
    z.add_registration_code('add_property("data", &::get_data)')
    mb.add_ndarray_interface(z)
    cc.write('''
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
            z2 = mb.class_(lambda x: x.alias=='vector_'+key)
            mb.asClass(z, z2, 'return convert_from_Mat_to_vector_of_T<%s>(inst)' % elem_type)
            mb.asClass(z2, z, 'return convert_from_vector_of_T_to_Mat<%s>(inst)' % elem_type)
        except RuntimeError:
            pass
        try:
            z2 = mb.class_(lambda x: x.alias=='vector_vector_'+key)
            mb.asClass(z, z2, 'return convert_from_Mat_to_vector_of_vector_of_T<%s>(inst)' % elem_type)
        except RuntimeError:
            pass
    mb.finalize_class(z)

    # rewrite the asMat function
    cc.write('''
def reshapeSingleChannel(mat):
    """Reshapes a Mat object into one that has a single channel.
    
    The function returns mat itself if it is single-channel.

    If it is multi-channel, the function invokes mat.reshape() to reshape
    the object. If the object has a single row, the returning object has
    rows=mat.cols and cols=mat.channels(). Otherwise, the returning object
    has rows=mat.rows and cols=mat.cols*mat.channels().    
    """
    if mat.channels() != 1:
        new_mat = mat.reshape(1, mat.cols if mat.rows==1 else mat.rows)
        if '_depends' in mat.__dict__:
            new_mat._depends = mat._depends
        return new_mat
    return mat
    
def asMat(obj, force_single_channel=False):
    """Converts a Python object into a Mat object.
    
    This general-purpose meta-function uses a simple heuristic method to
    identify the type of the given Python object in order to convert it into
    a Mat object. First, it tries to invoke the internal asMat() function of 
    the Python extension to convert. If not successful, it assumes the 
    object is a Python sequence, and converts the object into a std::vector 
    object whose element type is the type of the first element of the Python 
    sequence. After that, it converts the std::vector object into a Mat 
    object by invoking the internal asMat() function again.
    
    In the case that the above heuristic method does not convert into a Mat
    object with your intended type and depth, use one of the asvector_...()
    functions to convert your object into a vector before invoking asMat().
    
    If 'force_single_channel' is True, the returing Mat is single-channel (by
    invoking reshapeSingleChannel()). Otherwise, PyOpenCV tries to return a 
    multi-channel Mat whenever possible.
    """
    
    if obj is None:
        return Mat()
        
    try:
        out_mat = eval("_PE.asMat(inst_%s=obj)" % obj.__class__.__name__)
    except TypeError: # Boost.Python.ArgumentError is an unexposed subclass
        z = obj[0]
        if isinstance(z, int):
            out_mat = _PE.asMat(inst_vector_int=vector_int.fromlist(obj))
        elif isinstance(z, float):
            out_mat = _PE.asMat(inst_vector_float64=vector_float64.fromlist(obj))
        else:
            out_mat = eval("_PE.asMat(inst_vector_Type=vector_Type.fromlist(obj))"\
                .replace("Type", z.__class__.__name__))
    
    if force_single_channel:
        return reshapeSingleChannel(out_mat)
    return out_mat
    ''')
    
    # Ptr<Mat>
    mb.expose_class_Ptr('Mat', 'cv')
    common.register_vec('std::vector', 'cv::Ptr< cv::Mat >')    

    
    # Mat_
    # Minh-Tri: really bad idea to enable these classes, longer compilation 
    # time yet no real gain is observed
    # Mat_list = []
    # for suffix in "bswifd":
        # for i in xrange(1,5):
            # z = mb.class_(lambda x: x.alias=="Mat"+str(i)+suffix)
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

    # mb.dtypecast(Mat_list)

    # RNG
    z = mb.class_('RNG')
    mb.init_class(z)
    z.operator(lambda x: x.name.endswith('uchar')).rename('as_uint8')
    z.operator(lambda x: x.name.endswith('schar')).rename('as_int8')
    z.operator(lambda x: x.name.endswith('ushort')).rename('as_uint16')
    z.operator(lambda x: x.name.endswith('short int')).rename('as_int16')
    z.operator(lambda x: x.name.endswith('unsigned int')).rename('as_uint')
    z.operator(lambda x: x.name.endswith('operator int')).rename('as_int')
    z.operator(lambda x: x.name.endswith('float')).rename('as_float32')
    z.operator(lambda x: x.name.endswith('double')).rename('as_float64')
    z.mem_fun(lambda x: x.name=='uniform' and 'int' in x.partial_decl_string).rename('uniform_int')
    z.mem_fun(lambda x: x.name=='uniform' and 'float' in x.partial_decl_string).rename('uniform_float32')
    z.mem_fun(lambda x: x.name=='uniform' and 'double' in x.partial_decl_string).rename('uniform_float64')
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(state=" + repr(self.state) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    mb.finalize_class(z)
    
    # TermCriteria
    z = mb.class_('TermCriteria')
    mb.init_class(z)
    mb.decls(lambda x: 'CvTermCriteria' in x.partial_decl_string).exclude()
    # mb.asClass(z, mb.class_('CvTermCriteria'))
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(type=" + repr(self.type) + ", maxCount=" + repr(self.maxCount) + \\
        ", epsilon=" + repr(self.epsilon) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    mb.finalize_class(z)
    
    # PCA and SVD
    for t in ('::cv::PCA', '::cv::SVD'):
        z = mb.class_(t)
        mb.init_class(z)
        z.operator('()').call_policies = CP.return_self()
        mb.finalize_class(z)
        
    # LineIterator
    z = mb.class_(lambda x: x.name=='LineIterator' and x.parent.name=='sdopencv')
    mb.init_class(z)
    z.mem_fun('iter').rename('__iter__')
    mb.finalize_class(z)
    
    # MatND
    z = mb.class_('MatND')
    z.include_files.append("boost/python/make_function.hpp")
    z.include_files.append("opencv_converters.hpp")
    z.include_files.append("boost/python/str.hpp")
    mb.init_class(z)
    FT.set_array_item_type_as_size_t(z, 'step')
    common.register_vec('std::vector', 'cv::MatND')
    
    z.constructors(lambda x: 'const *' in x.decl_string).exclude()
    z.operator('()').exclude() # list of ranges, use ndarray instead
    z.add_declaration_code('''
static boost::shared_ptr<cv::MatND> MatND__init1__(cv::Mat const &_sizes, int _type)
{
    int* _sizes2; int _sizes3; convert_from_Mat_to_array_of_T(_sizes, _sizes2, _sizes3);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(_sizes3, _sizes2, _type));
}

static boost::shared_ptr<cv::MatND> MatND__init2__(cv::Mat const &_sizes, int _type, const cv::Scalar& _s)
{
    int* _sizes2; int _sizes3; convert_from_Mat_to_array_of_T(_sizes, _sizes2, _sizes3);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(_sizes3, _sizes2, _type, _s));
}

static boost::shared_ptr<cv::MatND> MatND__init3__(const cv::MatND& m, cv::Mat const &_ranges)
{
    cv::Range* _ranges2; int _ranges3; convert_from_Mat_to_array_of_T(_ranges, _ranges2, _ranges3);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(m, _ranges2));
}

static cv::MatND MatND__call__(const cv::MatND& inst, cv::Mat const &ranges)
{
    cv::Range* ranges2; int ranges3; convert_from_Mat_to_array_of_T(ranges, ranges2, ranges3);
    return inst(ranges2);
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init1__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type") )), "Use asMat() to convert \'_sizes\' from a Python sequence to a Mat.")')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init2__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type"), bp::arg("s") )), "Use asMat() to convert \'_sizes\' from a Python sequence to a Mat.")')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init3__, bp::default_call_policies(), ( bp::arg("m"), bp::arg("_ranges") )), "Use asMat() to convert \'_ranges\' from a Python sequence to a Mat.")')
    z.add_registration_code('def("__call__", bp::make_function(&MatND__call__, bp::default_call_policies(), (bp::arg("ranges"))), "Use asMat() to convert \'ranges\' from a Python sequence to a Mat.")')
    
    # mb.add_declaration_code('''
# struct CvMatND_to_python
# {
    # static PyObject* convert(CvMatND const& x)
    # {
        # return bp::incref(bp::object(cv::MatND(&x)).ptr());
    # }
# };

    # ''')
    # mb.add_registration_code('bp::to_python_converter<CvMatND, CvMatND_to_python, false>();')

    z.decls(lambda x: 'CvMatND' in x.decl_string).exclude()
    z.mem_funs('setTo').call_policies = CP.return_self()
    for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        z.decls(t).exclude()
    z.add_declaration_code('''
static bp::object get_data(cv::MatND const &inst)
{
    return bp::object(bp::handle<>(bp::borrowed(PyBuffer_FromReadWriteMemory(
        (void*)inst.data, inst.size[inst.dims-1]*inst.step[inst.dims-1]))));
}

    ''')
    z.add_registration_code('add_property("data", ::get_data)')
    mb.finalize_class(z)
    mb.add_ndarray_interface(z)
    cc.write('''
def _MatND__repr__(self):
    return "MatND(shape=" + repr(self.ndarray.shape) + ", nchannels=" + repr(self.channels()) \
        + ", depth=" + repr(self.depth()) + "):\\n" + repr(self.ndarray)
MatND.__repr__ = _MatND__repr__
    ''')
    

    # NAryMatNDIterator
    z = mb.class_('NAryMatNDIterator')
    mb.init_class(z)
    z.constructors(lambda x: "MatND const *" in x.partial_decl_string).exclude() # TODO: fix these constructors
    z.mem_fun('init')._transformer_creators.append(FT.input_as_list_of_MatND('arrays', 'count'))
    mb.finalize_class(z)
    
    # SparseMat
    # wait until requested: fix the rest of the member declarations
    z = mb.class_('SparseMat')
    z.include_files.append("opencv_converters.hpp")
    z.include_files.append("boost/python/make_function.hpp")
    mb.init_class(z)    
    z.constructors(lambda x: 'int const *' in x.decl_string).exclude()
    for t in ('CvSparseMat', 'Node', 'Hdr'):
        z.decls(lambda x: t in x.decl_string).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::SparseMat> SparseMat__init1__(cv::Mat const &_sizes, int _type)
{
    int* _sizes2; int _sizes3; convert_from_Mat_to_array_of_T(_sizes, _sizes2, _sizes3);
    return boost::shared_ptr<cv::SparseMat>(new cv::SparseMat(_sizes3, _sizes2, _type));
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&SparseMat__init1__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type") )))')    
    z.mem_funs('size').exclude()
    z.add_declaration_code('''
static bp::object my_size(cv::SparseMat const &inst, int i = -1)
{
    if(i >= 0) return bp::object(inst.size(i));
    
    bp::list l;
    const int *sz = inst.size();
    for(i = 0; i < inst.dims(); ++i) l.append(bp::object(sz[i]));
    return bp::tuple(l);
}
    ''')
    z.add_registration_code('def("size", (void (*)(int))(&my_size), (bp::arg("i")=bp::object(-1)))')
    z.mem_fun(lambda x: x.name == 'hash' and 'int const *' in x.arguments[0].type.decl_string) \
        ._transformer_creators.append(FT.input_array1d('idx'))
    for z2 in z.mem_funs('erase'):
        z2._transformer_creators.append(FT.output_type1('hashval'))
        if z2.arguments[0].name == 'idx':
            z2._transformer_creators.append(FT.input_array1d('idx'))
    for t in ('node', 'newNode', 'removeNode', 'hdr', 'ptr', 'begin', 'end'):
        z.decls(t).exclude()                
    mb.finalize_class(z)
    
    # SparseMatConstIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('SparseMatConstIterator')
    z.include()
    z.decls().exclude()
    
    # SparseMatIterator
    # TODO: fix the rest of the member declarations
    z = mb.class_('SparseMatIterator')
    z.include()
    z.decls().exclude()
    
    # KDTree
    # TODO: fix the rest of the member declarations
    z = mb.class_('KDTree')
    mb.init_class(z)
    common.register_vec('std::vector', 'cv::KDTree::Node', 'vector_KDTree_Node')
    z.decls().exclude()
    mb.finalize_class(z)
    
    # FileStorage
    z = mb.class_('FileStorage')
    mb.init_class(z)
    z.decls(lambda x: 'CvFileStorage' in x.decl_string).exclude()
    z.operators(lambda x: '*' in x.name or 'char' in x.decl_string).exclude()
    z.mem_fun('writeRaw')._transformer_creators.append(FT.input_array1d('vec', 'len'))
    z.mem_fun('writeObj').exclude() # too old
    for t in ('structs', 'fs'): # TODO: expose 'structs' but not 'fs'
        z.var(t).exclude()
    mb.finalize_class(z)
   
    # FileNode
    z = mb.class_('FileNode')
    z.include_files.append("opencv_converters.hpp")
    mb.init_class(z)
    z.decls(lambda x: 'CvFileStorage' in x.decl_string).exclude()
    z.operators(lambda x: '*' in x.name or 'char' in x.decl_string).exclude()
    z.operator(lambda x: x.name.endswith('operator int')).rename('as_int')
    z.operator(lambda x: x.name.endswith('float')).rename('as_float32')
    z.operator(lambda x: x.name.endswith('double')).rename('as_float64')
    z.operator(lambda x: x.name.endswith('string')).rename('as_str')
    for t in ('readObj', 'readRaw', 'begin', 'end', 'fs', 'node', 'rawDataSize'):
        z.decl(t).exclude()
    z.add_declaration_code('''
static bp::tuple children(cv::FileNode const &inst)
{
    bp::list l;
    for(cv::FileNodeIterator i = inst.begin(); i != inst.end(); ++i)
        l.append(bp::object(*i));
    return bp::tuple(l);
}

static cv::Mat readRaw(cv::FileNode const &inst, std::string const &fmt, int len)
{
    std::vector<uchar> data;
    data.resize(len);
    inst.readRaw(fmt, &data[0], len);
    return convert_from_vector_of_T_to_Mat<uchar>(data);
}

    ''')
    z.add_registration_code('add_property("children", &::children)')
    z.add_registration_code('def("__iter__", &::children)')
    z.mem_fun('readRaw').exclude()
    # wait until rawDataSize() is implemented
    z.add_registration_code('def("readRaw", &::readRaw, ( bp::arg("inst"), bp::arg("fmt"), bp::arg("len") ), "Reads raw data. Argument \'vec\' is returned as a Mat.")')
    mb.finalize_class(z)
    
    
    #=============================================================================
    # Free functions
    #=============================================================================
    

    # free functions
    for z in ('fromUtf16', 'toUtf16',
        'setNumThreads', 'getNumThreads', 'getThreadNum',
        'getTickCount', 'getTickFrequency', 'getCPUTickCount', 'checkHardwareSupport',
        'setUseOptimized', 'useOptimized',
        ):
        mb.free_fun(lambda decl: z in decl.name).include()
        
    # free functions
    for z in (
        'getElemSize',
        # 'cvarrToMat', 'extractImageCOI', 'insertImageCOI', # removed, deal with cv::Mat instead
        'add', 'subtract', 'multiply', 'divide', 'scaleAdd', 'addWeighted',
        'convertScaleAbs', 'LUT', 'sum', 'countNonZero', 'mean', 'meanStdDev', 
        'norm', 'normalize', 'reduce', 'flip', 'repeat', 'bitwise_and', 'bitwise_or', 
        'bitwise_xor', 'bitwise_not', 'absdiff', 'inRange', 'compare', 'cubeRoot', 
        'fastAtan2', 'polarToCart', 'cartToPolar', 'phase', 'magnitude', 'gemm',
        'mulTransposed', 'transpose', 'transform', 'perspectiveTransform',
        'completeSymm', 'setIdentity', 'determinant', 'trace', 'invert', 
        'solve', 'sort', 'sortIdx', 'eigen', 'Mahalanobis', 'Mahalonobis', 
        'dft', 'idft', 'dct', 'idct', 'mulSpectrums', 'getOptimalDFTSize',
        'randu', 'randn', 'randShuffle', 'line', 'rectangle', 'circle', 
        'ellipse', 'clipLine', 'putText', 'ellipse2Poly',
        ):
        mb.free_funs(z).include()

    for t in ('min', 'max', 'sqrt', 'pow', 'exp', 'log'):
        for z in mb.free_funs(t):
            if 'cv::Mat' in z.decl_string:
                z.include()

    # split, merge
    for t in ('split', 'merge'):
        for z in mb.free_funs(t):
            if 'vector' in z.partial_decl_string:
                z.include()
                # z._transformer_creators.append(FT.arg_std_vector('mv'))
                # z._transformer_kwds['alias'] = t
            
    # mixChannels
    for z in mb.free_funs('mixChannels'):
        if 'vector' in z.partial_decl_string:
            z.include()
            # z._transformer_creators.append(FT.arg_std_vector('src'))
            # z._transformer_creators.append(FT.arg_std_vector('dst'))
            z._transformer_kwds['alias'] = 'mixChannels'
            z._transformer_creators.append(FT.input_array1d('fromTo'))
    
    # minMaxLoc
    for z in mb.free_funs('minMaxLoc'):
        z.include()
        z._transformer_kwds['alias'] = 'minMaxLoc'
        for i in xrange(1,5):
            z._transformer_creators.append(FT.output_type1(i))
    
    # checkRange
    for z in mb.free_funs('checkRange'):
        z.include()
        z._transformer_creators.append(FT.output_type1(2))
        z._transformer_kwds['alias'] = 'checkRange'
    
    # kmeans
    z = mb.free_fun('kmeans')
    z.include()
    z._transformer_creators.append(FT.output_type1('centers'))
    
    # calcCovarMatrix
    for z in mb.free_funs('calcCovarMatrix'):
        z.include()
        if z.arguments[0].type == D.dummy_type_t('::cv::Mat const *'):
            z._transformer_creators.append(FT.input_array1d('samples', 'nsamples'))
        z._transformer_kwds['alias'] = 'calcCovarMatrix'
            
    # theRNG
    z = mb.free_fun('theRNG')
    z.include()
    z.call_policies = CP.return_value_policy(CP.reference_existing_object)
    
    # fillConvexPoly
    z = mb.free_fun('fillConvexPoly')
    z.include()
    z._transformer_creators.append(FT.input_array1d('pts', 'npts'))
    
    # fillPoly
    for t in ('fillPoly', 'polylines'):
        z = mb.free_fun(t)
        z.include()
        z._transformer_creators.append(FT.input_array2d('pts', 'npts', 'ncontours'))
        z._transformer_kwds['alias'] = t
        
    # getTextSize
    z = mb.free_fun('getTextSize')
    z.include()
    z._transformer_creators.append(FT.output_type1('baseLine'))

    # MemStorage
    common.register_ti('cv::Ptr', ['CvMemStorage'], 'MemStorage')
    mb.expose_class_Ptr('CvMemStorage')
    
    # Seq
    # TODO: do something with Seq<>

    # MatExpr
    mb.decls(lambda x: 'MatExpr' in x.decl_string).exclude()
    
