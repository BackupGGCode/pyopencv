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
    
    # Vec et al
    for suffix in dtype_dict:
        for i in xrange(2,5):
            mb.class_('::cv::Vec<%s, %i>' % (dtype_dict[suffix], i)).rename('Vec%d%s' % (i, suffix))
    zz = mb.classes(lambda z: z.name.startswith('Vec<'))
    for z in zz:
        z.include()
        mb.asClass(z, mb.class_('CvScalar'))
        z.decl('val').exclude() # use operator[] instead
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(" + self.ndarray.__str__() + ")"
KLASS.__repr__ = _KLASS__repr__
        '''.replace('KLASS', z.alias))
    mb.dtypecast(['::cv::Vec<%s, 2>' % x \
        for x in ['unsigned char', 'short', 'unsigned short', 'int', 'float', 'double']])
    mb.dtypecast(['::cv::Vec<%s, 3>' % x \
        for x in ['unsigned char', 'short', 'unsigned short', 'int', 'float', 'double']])
    mb.dtypecast(['::cv::Vec<%s, 4>' % x \
        for x in ['unsigned char', 'short', 'unsigned short', 'int', 'float', 'double']])
    mb.dtypecast(['::cv::Vec<%s, 6>' % x for x in ['float', 'double']])
        
    # Complex et al
    zz = mb.classes(lambda z: z.name.startswith('Complex<'))
    for z in zz:
        z.include()
        z.decls(lambda t: 'std::complex' in t.decl_string).exclude() # no std::complex please
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(re=" + repr(self.re) + ", im=" + repr(self.im) + ")"
KLASS.__repr__ = _KLASS__repr__
        '''.replace('KLASS', z.alias))
    mb.dtypecast(['::cv::Complex<%s>' % x for x in ['float', 'double']])
    
    # Point et al
    mb.class_('::cv::Point_<int>').rename('Point2i')
    zz = mb.classes(lambda z: z.name.startswith('Point_<'))
    for z in zz:
        z.include()
        mb.asClass(z, mb.class_('CvPoint'))
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.add_ndarray_interface(z)
    
    cc.write('''
Point = Point2i
asPoint = asPoint2i
    ''')
    mb.dtypecast(['::cv::Point_<%s>' % x for x in ['int', 'float', 'double']])
    
    # Point3 et al
    mb.class_('::cv::Point3_<float>').rename('Point3f')
    zz = mb.classes(lambda z: z.name.startswith('Point3_<'))
    for z in zz:
        z.include()
        mb.asClass(z, mb.class_('CvPoint3D32f'))
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ", z=" + repr(self.z) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    mb.dtypecast(['::cv::Point3_<%s>' % x for x in ['int', 'float', 'double']])
    
    # Size et al
    mb.class_('::cv::Size_<int>').rename('Size2i')
    zz = mb.classes(lambda z: z.name.startswith('Size_<'))
    for z in zz:
        z.include()
        mb.asClass(z, mb.class_('CvSize'))
        mb.asClass(z, mb.class_('CvSize2D32f'))
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    mb.dtypecast(['::cv::Size_<%s>' % x for x in ['int', 'float', 'double']])
        
    cc.write('''
Size = Size2i
    ''')
    
    # Rect et al
    zz = mb.classes(lambda z: z.name.startswith('Rect_<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvRect' in x.decl_string).exclude()
        mb.add_ndarray_interface(z)
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + \\
        ", width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    mb.dtypecast(['::cv::Rect_<%s>' % x for x in ['int', 'float', 'double']])
    
    # RotatedRect
    z = mb.class_('RotatedRect')
    z.include()
    mb.asClass(z, mb.class_('CvBox2D'))
    mb.add_ndarray_interface(z)
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(center=" + repr(self.center) + ", size=" + repr(self.size) + \\
        ", angle=" + repr(self.angle) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # Scalar et al
    zz = mb.classes(lambda z: z.name.startswith('Scalar_<'))
    for z in zz:
        z.include()
        mb.asClass(z, mb.class_('CvScalar'))
    z = mb.class_('::cv::Scalar_<double>')
    z.rename('Scalar')    
    mb.add_ndarray_interface(z)
    cc.write('''
def _Scalar__repr__(self):
    return "Scalar(" + self.ndarray.__str__() + ")"
Scalar.__repr__ = _Scalar__repr__
    ''')
    
    # Range
    z = mb.class_('Range')
    z.include()
    mb.asClass(z, mb.class_('CvSlice'))
    mb.add_ndarray_interface(z)
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(start=" + repr(self.start) + ", end=" + repr(self.end) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # Ptr -- already exposed by mb.expose_class_Ptr
    
    # Mat
    z = mb.class_('Mat')
    z.include_files.append("opencv_converters.hpp")
    z.include()
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
    return bp::object(bp::handle<>(PyBuffer_FromReadWriteMemory ((void*)inst.data, inst.rows*inst.step)));
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
        'Rectf': 'cv::Rectf',
        'Rectd': 'cv::Rectd',
        'RotatedRect': 'cv::RotatedRect',
        'Size2i': 'cv::Size2i',
        'Size2f': 'cv::Size2f',
        'Size2d': 'cv::Size2d',
        'Scalar': 'cv::Scalar',
        'Range': 'cv::Range',
    }
    for key in list_dict:
        z.add_registration_code('def("to_list_of_%s", &convert_from_Mat_to_seq<%s> )' % (key, list_dict[key]))
        z.add_registration_code('def("from_list_of_%s", &convert_from_seq_to_Mat_object<%s> )' % (key, list_dict[key]))
        z.add_registration_code('staticmethod("from_list_of_%s")' % key)
    # rewrite the asMat function
    cc.write('''
def asMat(obj, force_single_channel=False):
    """Converts a Python object into a Mat object.
    
    If 'force_single_channel' is True, the returing Mat is single-channel. Otherwise, PyOpenCV tries to return a multi-channel Mat whenever possible.
    """
    
    if obj is None:
        return Mat()
    
    if isinstance(obj, _NP.ndarray):
        out_mat = Mat.from_ndarray(obj)
    else:
        z = obj[0]
        if isinstance(z, int):
            out_mat = Mat.from_list_of_int(obj)
        elif isinstance(z, float):
            out_mat = Mat.from_list_of_float64(obj)
        else:
            out_mat = eval("Mat.from_list_of_%s(obj)" % z.__class__.__name__)
    
    if force_single_channel and out_mat.channels() != 1:
        return out_mat.reshape(1, out_mat.cols if out_mat.rows==1 else out_mat.rows)
        
    return out_mat
    ''')
    
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
    z.include()
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
    
    # TermCriteria
    z = mb.class_('TermCriteria')
    z.include()
    mb.asClass(z, mb.class_('CvTermCriteria'))
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(type=" + repr(self.type) + ", maxCount=" + repr(self.maxCount) + \\
        ", epsilon=" + repr(self.epsilon) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # PCA and SVD
    for t in ('::cv::PCA', '::cv::SVD'):
        z = mb.class_(t)
        mb.init_class(z)
        z.operator('()').call_policies = CP.return_self()
        mb.finalize_class(z)
        
    # LineIterator
    z = mb.class_('LineIterator')
    z.include_files.append("sdopencv/dtype.hpp")
    z.include()
    z.operator('*').exclude()
    z.var('ptr').exclude()
    # replace operator*() with 'get_pixel_addr', not the best solution, if you have a better one, send me a patch
    z.add_declaration_code('static sdopencv::address_t get_pixel_addr(cv::LineIterator &inst) { return (sdopencv::address_t)(*inst); }')
    z.add_registration_code('def("get_pixel_addr", &get_pixel_addr)')
    # replace operator++() with 'inc'
    z.operators('++').exclude()
    z.add_declaration_code('static cv::LineIterator & inc(cv::LineIterator &inst) { return ++inst; }')
    z.add_registration_code('def("inc", bp::make_function(&inc, bp::return_self<>()) )')
    
    # MatND
    z = mb.class_('MatND')
    z.include_files.append("boost/python/make_function.hpp")
    z.include_files.append("opencv_converters.hpp")
    z.include_files.append("boost/python/str.hpp")
    mb.init_class(z)
    
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
    return bp::object(bp::handle<>(PyBuffer_FromReadWriteMemory ((void*)inst.data, inst.size[inst.dims-1]*inst.step[inst.dims-1])));
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
    for t in ('arrays', 'planes'): # TODO: expose these variables of type std::vector<Mat..>
        z.var(t).exclude()
    z.mem_fun('init')._transformer_creators.append(FT.input_as_list_of_MatND('arrays', 'count'))
    mb.finalize_class(z)
    
    # SparseMat
    # wait until requested: fix the rest of the member declarations
    z = mb.class_('SparseMat')
    z.include_files.append("opencv_converters.hpp")
    z.include()
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
    z.include()
    z.decls().exclude()
    
    # FileStorage
    # TODO: wrap writeRaw and writeObj
    z = mb.class_('FileStorage')
    z.include()
    z.constructor(lambda x: 'CvFileStorage' in x.decl_string).exclude()
    z.operators(lambda x: 'char' in x.decl_string).exclude()
    z.operators('*').exclude()
    for t in ('writeRaw', 'writeObj'):
        z.decl(t).exclude()
    mb.expose_class_Ptr('CvFileStorage')
   
    # FileNode
    # TODO: wrap readRaw and readObj, and fix the problem with operator float and double at the same time
    z = mb.class_('FileNode')
    z.include()
    z.constructors(lambda x: len(x.arguments)==2).exclude()
    z.operators(lambda x: 'char' in x.decl_string).exclude()
    z.operators('*').exclude()
    z.mem_fun('rawDataSize').exclude() # missing function    
    for t in ('readRaw', 'readObj', 'fs', 'node', 'begin', 'end'):
        z.decl(t).exclude()
    z.add_declaration_code('''
static bp::tuple children(cv::FileNode const &inst)
{
    bp::list l;
    for(cv::FileNodeIterator i = inst.begin(); i != inst.end(); ++i)
        l.append(bp::object(*i));
    return bp::tuple(l);
}
    ''')
    z.add_registration_code('def("children", &::children)')
    
    
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
                z._transformer_creators.append(FT.arg_std_vector('mv'))
                z._transformer_kwds['alias'] = t
            
    # mixChannels
    for z in mb.free_funs('mixChannels'):
        if 'vector' in z.partial_decl_string:
            z.include()
            z._transformer_creators.append(FT.arg_std_vector('src'))
            z._transformer_creators.append(FT.arg_std_vector('dst'))
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
    
    # TODO: do something with Seq<>

    # MatExpr
    mb.decls(lambda x: 'MatExpr' in x.decl_string).exclude()
    
