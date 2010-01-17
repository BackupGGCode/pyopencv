#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.0 using Boost.Python and NumPy

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

try:
    Size = Size2i
except:
    Size2i = Size

Point = Point2i

    ''')
    
    #=============================================================================
    # Structures
    #=============================================================================
    
    # Vec et al
    mb.class_('::cv::Vec<int, 4>').rename('Vec4i')
    zz = mb.classes(lambda z: z.name.startswith('Vec<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvScalar' in x.decl_string).exclude()
        z.decl('val').exclude() # use operator[] instead
        
    # Complex et al
    zz = mb.classes(lambda z: z.name.startswith('Complex<'))
    for z in zz:
        z.include()
        z.decls(lambda t: 'std::complex' in t.decl_string).exclude() # no std::complex please
    
    # Point et al
    mb.class_('::cv::Point_<int>').rename('Point2i')
    zz = mb.classes(lambda z: z.name.startswith('Point_<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvPoint' in x.decl_string).exclude()
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.add_ndarray_interface(z)
    
    # Point3 et al
    mb.class_('::cv::Point3_<float>').rename('Point3f')
    zz = mb.classes(lambda z: z.name.startswith('Point3_<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvPoint' in x.decl_string).exclude()
        z.operator(lambda x: '::cv::Vec<' in x.name).rename('as_Vec'+z.alias[-2:])
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + ", z=" + repr(self.z) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
        mb.add_ndarray_interface(z)
    
    # Size et al
    mb.class_('::cv::Size_<int>').rename('Size2i')
    zz = mb.classes(lambda z: z.name.startswith('Size_<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvSize' in x.decl_string).exclude()
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    
    # Rect et al
    zz = mb.classes(lambda z: z.name.startswith('Rect_<'))
    for z in zz:
        z.include()
        z.decls(lambda x: 'CvRect' in x.decl_string).exclude()
        cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + \\
        ", width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
        '''.replace("KLASS", z.alias))
    
    # RotatedRect
    z = mb.class_('RotatedRect')
    z.include()
    z.decls(lambda x: 'CvBox2D' in x.decl_string).exclude()
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
        z.decls(lambda x: 'CvScalar' in x.decl_string).exclude()
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
    z.operator(lambda x: x.name.endswith('::CvSlice')).rename('as_CvSlice')
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(start=" + repr(self.start) + ", end=" + repr(self.end) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # Ptr -- already exposed by mb.expose_class_Ptr

    # Mat
    z = mb.class_('Mat')
    z.include_files.append("opencv_extra.hpp")
    z.include()
    for t in ('::IplImage', '::CvMat', 'MatExp'):
        z.decls(lambda x: t in x.decl_string).exclude()
    z.mem_funs('setTo').call_policies = CP.return_self()
    z.mem_funs('adjustROI').call_policies = CP.return_self()
    for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        z.decls(t).exclude()
    mb.add_ndarray_interface(z)
    cc.write('''
def _Mat__repr__(self):
    return "Mat()" if self.empty() else "Mat(rows=" + repr(self.rows) \
        + ", cols=" + repr(self.cols) + ", nchannels=" + repr(self.channels()) \
        + ", depth=" + repr(self.depth()) + "):\\n" + repr(self.ndarray)
Mat.__repr__ = _Mat__repr__
    ''')
    z.add_declaration_code('''
static boost::shared_ptr<cv::Mat> Mat__init1__(const bp::object &seq)
{
    cv::Mat *result = new cv::Mat();
    convert_Mat(seq, *result);
    return boost::shared_ptr<cv::Mat>(result);
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&Mat__init1__, bp::default_call_policies(), ( bp::arg("seq") )))')

    # RNG
    z = mb.class_('RNG')
    z.include()
    z.operator(lambda x: x.name.endswith('uchar')).rename('as_uchar')
    z.operator(lambda x: x.name.endswith('schar')).rename('as_schar')
    z.operator(lambda x: x.name.endswith('ushort')).rename('as_ushort')
    z.operator(lambda x: x.name.endswith('short int')).rename('as_short')
    z.operator(lambda x: x.name.endswith('unsigned int')).rename('as_unsigned')
    z.operator(lambda x: x.name.endswith('operator int')).rename('as_int')
    z.operator(lambda x: x.name.endswith('float')).rename('as_float')
    z.operator(lambda x: x.name.endswith('double')).rename('as_double')
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(state=" + repr(self.state) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # TermCriteria
    z = mb.class_('TermCriteria')
    z.include()
    # z.decls(lambda x: 'CvTermCriteria' in x.decl_string).exclude()
    cc.write('''
def _KLASS__repr__(self):
    return "KLASS(type=" + repr(self.type) + ", maxCount=" + repr(self.maxCount) + \\
        ", epsilon=" + repr(self.epsilon) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    
    # PCA and SVD
    for t in ('::cv::PCA', '::cv::SVD'):
        z = mb.class_(t)
        z.include()
        z.operator('()').call_policies = CP.return_self()
        
    # LineIterator
    z = mb.class_('LineIterator')
    z.include()
    z.operator('*').exclude()
    z.var('ptr').exclude()
    # replace operator*() with 'get_pixel_addr', not the best solution, if you have a better one, send me a patch
    z.add_declaration_code('static int get_pixel_addr(cv::LineIterator &inst) { return (int)(*inst); }')
    z.add_registration_code('def("get_pixel_addr", &get_pixel_addr)')
    # replace operator++() with 'inc'
    z.operators('++').exclude()
    z.add_declaration_code('static cv::LineIterator & inc(cv::LineIterator &inst) { return ++inst; }')
    z.add_registration_code('def("inc", bp::make_function(&inc, bp::return_self<>()) )')
    
    # MatND
    z = mb.class_('MatND')
    z.include_files.append("boost/python/make_function.hpp")
    z.include_files.append("opencv_extra.hpp")
    mb.init_class(z)
    
    z.constructors(lambda x: 'const *' in x.decl_string).exclude()
    z.operator('()').exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::MatND> MatND__init1__(const bp::tuple &_sizes, int _type)
{
    std::vector<int> _sizes2;
    int len = bp::len(_sizes);
    _sizes2.resize(len);
    for(int i = 0; i < len; ++i) _sizes2[i] = bp::extract<int>(_sizes[i]);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(len, &_sizes2[0], _type));
}

static boost::shared_ptr<cv::MatND> MatND__init2__(const bp::tuple &_sizes, int _type, const cv::Scalar& _s)
{
    std::vector<int> _sizes2;
    int len = bp::len(_sizes);
    _sizes2.resize(len);
    for(int i = 0; i < len; ++i) _sizes2[i] = bp::extract<int>(_sizes[i]);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(len, &_sizes2[0], _type, _s));
}

static boost::shared_ptr<cv::MatND> MatND__init3__(const cv::MatND& m, const bp::tuple &_ranges)
{
    std::vector<cv::Range> _ranges2;
    int len = bp::len(_ranges);
    _ranges2.resize(len);
    for(int i = 0; i < len; ++i) _ranges2[i] = bp::extract<cv::Range>(_ranges[i]);
    return boost::shared_ptr<cv::MatND>(new cv::MatND(m, &_ranges2[0]));
}

static cv::MatND MatND__call__(const cv::MatND& inst, const bp::tuple &ranges)
{
    std::vector<cv::Range> ranges2;
    int len = bp::len(ranges);
    ranges2.resize(len);
    for(int i = 0; i < len; ++i) ranges2[i] = bp::extract<cv::Range>(ranges[i]);
    return inst(&ranges2[0]);
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init1__))')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init2__))')
    z.add_registration_code('def("__init__", bp::make_constructor(&MatND__init3__))')
    z.add_registration_code('def("__call__", bp::make_function(&MatND__call__))')
    
    mb.add_declaration_code('''
struct CvMatND_to_python
{
    static PyObject* convert(CvMatND const& x)
    {
        return bp::incref(bp::object(cv::MatND(&x)).ptr());
    }
};

    ''')
    mb.add_registration_code('bp::to_python_converter<CvMatND, CvMatND_to_python, false>();')

    z.decls(lambda x: 'CvMatND' in x.decl_string).exclude()
    z.mem_funs('setTo').call_policies = CP.return_self()
    for t in ('ptr', 'data', 'refcount', 'datastart', 'dataend'):
        z.decls(t).exclude()
    mb.finalize_class(z)
    mb.add_ndarray_interface(z)
    cc.write('''
def _MatND__repr__(self):
    return "MatND(shape=" + repr(self.ndarray.shape) + ", nchannels=" + repr(self.channels()) \
        + ", depth=" + repr(self.depth()) + "):\\n" + repr(self.ndarray)
MatND.__repr__ = _MatND__repr__
    ''')
    

    # NAryMatNDIterator
    # wait until requested: fix the rest of the member declarations
    z = mb.class_('NAryMatNDIterator')
    z.include()
    z.decls().exclude()
    
    # SparseMat
    # wait until requested: fix the rest of the member declarations
    z = mb.class_('SparseMat')
    z.include()
    z.include_files.append("boost/python/make_function.hpp")
    mb.init_class(z)
    
    z.constructors(lambda x: 'int const *' in x.decl_string).exclude()
    for t in ('CvSparseMat', 'Node', 'Hdr'):
        z.decls(lambda x: t in x.decl_string).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<cv::SparseMat> SparseMat__init1__(const bp::tuple &_sizes, int _type)
{
    std::vector<int> _sizes2;
    int len = bp::len(_sizes);
    _sizes2.resize(len);
    for(int i = 0; i < len; ++i) _sizes2[i] = bp::extract<int>(_sizes[i]);
    return boost::shared_ptr<cv::SparseMat>(new cv::SparseMat(len, &_sizes2[0], _type));
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&SparseMat__init1__))')
    
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
        'getTickCount', 'getTickFrequency',
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
        'bitwise_xor', 'bitwise_not', 'absdiff', 'inRange', 'compare', 'min', 
        'max', 'sqrt', 'pow', 'exp', 'log', 'cubeRoot', 'fastAtan2',
        'polarToCart', 'cartToPolar', 'phase', 'magnitude', 'gemm',
        'mulTransposed', 'transpose', 'transform', 'perspectiveTransform',
        'completeSymm', 'setIdentity', 'determinant', 'trace', 'invert', 
        'solve', 'sort', 'sortIdx', 'eigen', 'Mahalanobis', 'Mahalonobis', 
        'dft', 'idft', 'dct', 'idct', 'mulSpectrums', 'getOptimalDFTSize',
        'randu', 'randn', 'randShuffle', 'line', 'rectangle', 'circle', 
        'ellipse', 'clipLine', 'putText', 'ellipse2Poly',
        ):
        mb.free_funs(z).include()

    # split
    for z in mb.free_funs('split'):
        if z.arguments[1].type == D.dummy_type_t('::cv::Mat *') or \
            z.arguments[1].type == D.dummy_type_t('::cv::MatND *'):
            z.include()
            z._transformer_creators.append(FT.input_array1d(1))
            z._transformer_kwds['alias'] = 'split'
    
    # mixChannels
    z = mb.free_funs('mixChannels').exclude()
    mb.add_registration_code('bp::def("mixChannels", &bp::mixChannels, ( bp::arg("src"), bp::arg("dst"), bp::arg("fromTo") ));')
    
    # minMaxLoc
    z = mb.free_funs('minMaxLoc').exclude()
    mb.add_registration_code('bp::def("minMaxLoc", &bp::minMaxLoc, ( bp::arg("a"), bp::arg("mask")=bp::object() ));')
    
    # checkRange
    for z in mb.free_funs('checkRange'):
        z.include()
        z._transformer_creators.append(FT.output_type1(2))
        z._transformer_kwds['alias'] = 'checkRange'
    
    # kmeans
    z = mb.free_fun('kmeans')
    z.include()
    z._transformer_creators.append(FT.output_type1('centers'))
    
    # merge
    for z in mb.free_funs('merge'):
        if z.arguments[0].type == D.dummy_type_t('::cv::Mat const *') or \
            z.arguments[0].type == D.dummy_type_t('::cv::MatND const *'):
            z.include()
            z._transformer_creators.append(FT.input_array1d(0, 'count'))
            z._transformer_kwds['alias'] = 'merge'
            
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
