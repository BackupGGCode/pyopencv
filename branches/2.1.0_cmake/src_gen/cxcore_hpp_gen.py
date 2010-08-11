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
sb = sdpypp.SdModuleBuilder('cxcore_hpp', number_of_files=4)
sb.load_regs('cxcore_hpp_point_reg.sdd')

sb.cc.write('''
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

Point_dict = 'ifd'

# Size et al
Size_dict = 'if'
for suffix in Size_dict:
    alias = 'Size2%s' % suffix
    sb.register_ti('cv::Size_', [dtype_dict[suffix]], alias)
    try:
        z = sb.mb.class_(lambda x: x.alias==alias)
    except RuntimeError:
        continue
    sb.init_class(z)
    sb.expose_class_vector(z.partial_decl_string[2:])
    z.decls(lambda x: 'CvSize' in x.partial_decl_string).exclude()
    # sb.asClass(z, sb.mb.class_('CvSize'))
    # sb.asClass(z, sb.mb.class_('CvSize2D32f'))
    sb.add_ndarray_interface(z)
    sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
    sb.finalize_class(z)
    
    # operations
    c = z
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
    
sb.dtypecast(['::cv::Size_<%s>' % dtype_dict[suffix] for suffix in Size_dict])
    
sb.cc.write('''
Size = Size2i
''')

# Rect
sb.register_ti('cv::Rect_', ['int'], 'Rect')
z = sb.mb.class_(lambda x: x.alias=='Rect')
sb.init_class(z)
sb.expose_class_vector(z.partial_decl_string[2:])
z.decls(lambda x: 'CvRect' in x.partial_decl_string).exclude()
# sb.asClass(z, sb.mb.class_('CvRect'))
sb.add_ndarray_interface(z)
sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(x=" + repr(self.x) + ", y=" + repr(self.y) + \\
        ", width=" + repr(self.width) + ", height=" + repr(self.height) + ")"
KLASS.__repr__ = _KLASS__repr__
        
    '''.replace("KLASS", z.alias))
sb.finalize_class(z)

# operations
dtype = 'int'
c = z
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

# sb.dtypecast(['::cv::Rect_<%s>' % dtype_dict[suffix] for suffix in Point_dict])

# RotatedRect
z = sb.mb.class_('RotatedRect')
sb.init_class(z)
sb.expose_class_vector(z.partial_decl_string[2:])
sb.mb.decls(lambda x: 'CvBox2D' in x.partial_decl_string).exclude()
# sb.asClass(z, sb.mb.class_('CvBox2D'))
sb.add_ndarray_interface(z)
sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(center=" + repr(self.center) + ", size=" + repr(self.size) + \\
        ", angle=" + repr(self.angle) + ")"
KLASS.__repr__ = _KLASS__repr__
        
'''.replace("KLASS", z.alias))
sb.finalize_class(z)

# Scalar et al
sb.register_ti('cv::Scalar_', ['double'], 'Scalar')
sb.cc.write('''
# Constructs a color value
def CV_RGB(r, g, b):
    return Scalar(b, g, r)

''')

# Range
sb.register_ti('cv::Range')
sb.register_vec('std::vector', 'cv::Range')


# Ptr -- already exposed by sb.expose_class_Ptr

# Mat
sb.register_ti('cv::Mat')

# Ptr<Mat>
sb.register_ti('cv::Ptr', ['cv::Mat'], 'Ptr_Mat')
sb.register_vec('std::vector', 'cv::Ptr< cv::Mat >')

# RNG
z = sb.mb.class_('RNG')
sb.init_class(z)
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
sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(state=" + repr(self.state) + ")"
KLASS.__repr__ = _KLASS__repr__
    
'''.replace("KLASS", z.alias))
sb.finalize_class(z)

# TermCriteria
z = sb.mb.class_('TermCriteria')
sb.init_class(z)
sb.mb.decls(lambda x: 'CvTermCriteria' in x.partial_decl_string).exclude()
# sb.asClass(z, sb.mb.class_('CvTermCriteria'))
sb.cc.write('''
def _KLASS__repr__(self):
    return "KLASS(type=" + repr(self.type) + ", maxCount=" + repr(self.maxCount) + \\
        ", epsilon=" + repr(self.epsilon) + ")"
KLASS.__repr__ = _KLASS__repr__
    
'''.replace("KLASS", z.alias))
sb.finalize_class(z)

# PCA and SVD
for t in ('::cv::PCA', '::cv::SVD'):
    z = sb.mb.class_(t)
    sb.init_class(z)
    z.operator('()').call_policies = CP.return_self()
    sb.finalize_class(z)
    
# LineIterator
z = sb.mb.class_('LineIterator')
sb.init_class(z)
z.constructors().exclude()
z.operators().exclude()
z.var('ptr').exclude()
z.add_wrapper_code('''
private:
    int iteration;
    int ws, es;
    uchar *ptr0;

public:
    LineIterator_wrapper(const cv::Mat& img, cv::Point const &pt1, cv::Point const &pt2,
        int connectivity=8, bool leftToRight=false)
        : cv::LineIterator(img, pt1, pt2, connectivity, leftToRight),
        iteration(0), ptr0(img.data), ws(img.step), es(img.elemSize()) {}
        
    LineIterator_wrapper const &iter() { return *this; }
    
    cv::Point next()
    {
        int ofs = (int)(ptr-ptr0);
        
        if(iteration < count)
        {
            ++(*this);
            ++iteration;
        }
        else
        {
            PyErr_SetString(PyExc_StopIteration, "No more pixel.");
            throw bp::error_already_set(); 
        }
        
        return cv::Point((ofs%ws)/es, ofs/ws);
    }

''')
z.add_registration_code('def(bp::init<cv::Mat const &, cv::Point const &, cv::Point const &, int, bool>(( bp::arg("img"), bp::arg("pt1"), bp::arg("pt2"), bp::arg("connectivity")=8, bp::arg("leftToRight")=false)))')
z.add_registration_code('def("__iter__", &::LineIterator_wrapper::iter, bp::return_self<>())')
z.add_registration_code('def("next", &::LineIterator_wrapper::next)')
sb.finalize_class(z)

# MatND
sb.register_ti('cv::MatND')

# NAryMatNDIterator
z = sb.mb.class_('NAryMatNDIterator')
sb.init_class(z)
z.constructors(lambda x: "MatND const *" in x.partial_decl_string).exclude() # don't need them
z.add_declaration_code('''
static boost::shared_ptr<cv::NAryMatNDIterator> NAryMatNDIterator__init1__(std::vector<cv::MatND> const &arrays)
{
    std::vector<cv::MatND const *> buf_arrays(arrays.size());
    for(size_t i_arrays = 0; i_arrays<arrays.size(); ++i_arrays)
        buf_arrays[i_arrays] = (cv::MatND const *)&(arrays[i_arrays]);
        
    return boost::shared_ptr<cv::NAryMatNDIterator>(new cv::NAryMatNDIterator((cv::MatND const * *)(&buf_arrays[0]), arrays.size()));
}

''')    
z.add_registration_code('def("__init__", bp::make_constructor(&NAryMatNDIterator__init1__, bp::default_call_policies(), (bp::arg("arrays"))))')
z.mem_fun('init')._transformer_creators.append(FT.input_as_list_of_Matlike('arrays', 'count'))
z.add_wrapper_code('''    
    NAryMatNDIterator_wrapper const &iter() { return *this; }
    
    bp::object next()
    {
        if(idx >= nplanes)
        {
            PyErr_SetString(PyExc_StopIteration, "No more plane.");
            throw bp::error_already_set(); 
        }
        bp::object result(planes);
        if(idx >= nplanes-1) ++idx;
        else ++(*this);
        return result;
    }

''')    
z.add_registration_code('def("__iter__", &NAryMatNDIterator_wrapper::iter, bp::return_self<>())')
z.add_registration_code('def("next", &NAryMatNDIterator_wrapper::next)')
sb.finalize_class(z)

# SparseMat
z = sb.mb.class_('SparseMat')
z.include_files.append("opencv_converters.hpp")
z.include_files.append("boost/python/make_function.hpp")
sb.init_class(z)    
z.constructors(lambda x: 'int const *' in x.decl_string).exclude()
# TODO: don't know how to expose the output value of newNode()
for t in ('addref', 'release', 'newNode', 'ptr', 'begin', 'end'):
    z.decls(t).exclude()
for t in ('CvSparseMat',):
    z.decls(lambda x: t in x.decl_string).exclude()
z.add_declaration_code('''
static boost::shared_ptr<cv::SparseMat> SparseMat__init1__(std::vector<int> const &_sizes, int _type)
{
    return boost::shared_ptr<cv::SparseMat>(new cv::SparseMat(_sizes.size(), &_sizes[0], _type));
}

''')
z.add_registration_code('def("__init__", bp::make_constructor(&SparseMat__init1__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type") )))')    
z.mem_funs('size').exclude()
z.add_declaration_code('''
static bp::object SparseMat_size(cv::SparseMat const &inst, int i = -1)
{
    if(i >= 0) return bp::object(inst.size(i));
    
    const int *sz = inst.size();
    if(!sz) return bp::object();
    
    std::vector<int> result(inst.dims());
    for(i = 0; i < inst.dims(); ++i) result[i] = sz[i];
    return bp::object(result);
}
''')
z.add_registration_code('def("size", &::SparseMat_size, (bp::arg("i")=bp::object(-1)))')
z1 = z.mem_fun(lambda x: x.name=="begin" and "const" not in x.partial_decl_string)
z1.include()
z1.rename("__iter__")
z.mem_fun(lambda x: x.name == 'hash' and 'int const *' in x.arguments[0].type.decl_string) \
    ._transformer_creators.append(FT.input_array1d('idx'))
for z2 in z.mem_funs('erase'):
    z2._transformer_creators.append(FT.output_type1('hashval'))
    if z2.arguments[0].name == 'idx':
        z2._transformer_creators.append(FT.input_array1d('idx'))
# Hdr
z1 = z.class_('Hdr')
sb.init_class(z1)
z1.constructor(lambda x: len(x.arguments) > 1).exclude()
z1.add_declaration_code('''
static boost::shared_ptr<cv::SparseMat::Hdr> SparseMat_Hdr__init1__(std::vector<int> const &_sizes, int _type)
{
    return boost::shared_ptr<cv::SparseMat::Hdr>(new cv::SparseMat::Hdr(_sizes.size(), &_sizes[0], _type));
}

''')
z1.add_registration_code('def("__init__", bp::make_constructor(&SparseMat_Hdr__init1__, bp::default_call_policies(), ( bp::arg("_sizes"), bp::arg("_type") )))')    
sb.finalize_class(z1)
sb.finalize_class(z)

# SparseMatConstIterator
sb.mb.class_('SparseMatConstIterator').exclude() # we don't want this class exposed

# SparseMatIterator
z = sb.mb.class_('SparseMatIterator')
sb.init_class(z)
z.constructors(lambda x: len(x.arguments) > 1).exclude()
z.add_wrapper_code('''    
    SparseMatIterator_wrapper const &iter() { return *this; }
    
    cv::SparseMat::Node *next()
    {
        if(!m || !ptr || !m->hdr)
        {
            PyErr_SetString(PyExc_StopIteration, "No more node.");
            throw bp::error_already_set(); 
        }
        
        cv::SparseMat::Node *result = node();
        ++(*this);
        return result;
    }

''')    
z.add_registration_code('def("__iter__", &SparseMatIterator_wrapper::iter, bp::return_self<>())')
z.add_registration_code('def("next", &SparseMatIterator_wrapper::next, bp::return_internal_reference<>())')
sb.finalize_class(z)

# KDTree
z = sb.mb.class_('KDTree')
z.include_files.append( "boost/python/object/life_support.hpp" )
z.include_files.append( "arrayobject.h" ) # to get NumPy's flags
z.include_files.append( "ndarray.hpp" )
sb.init_class(z)
sb.register_decl('KDTree_Node', 'cv::KDTree::Node')
sb.expose_class_vector('cv::KDTree::Node')
# dims -- OpenCV 2.1 does not have this function implemented!
z.add_declaration_code('''
inline int cv::KDTree::dims() const { return points.cols; }
''')
# findNearest
for t in z.mem_funs('findNearest'):
    if not 'vector' in t.partial_decl_string:
        t.exclude()
    t._transformer_creators.extend([FT.input_array1d('vec'), FT.output_type1('neighborsIdx'),
        FT.output_type1('neighbors'), FT.output_type1('dist')])
    t._transformer_kwds['alias'] = 'findNearest'
# findOrthoRange
z.mem_fun('findOrthoRange')._transformer_creators.extend([
    FT.input_array1d('minBounds'), FT.input_array1d('maxBounds'),
    FT.output_type1('neighborsIdx'), FT.output_type1('neighbors')])
# getPoints
for t in z.mem_funs('getPoints'):
    if t.arguments[0].name=='idx':
        t._transformer_creators.append(FT.input_array1d('idx', 'nidx'))
    t._transformer_creators.append(FT.arg_output('pts'))
    t._transformer_kwds['alias'] = 'getPoints'
# getPoint
z.mem_fun('getPoint').exclude()
# get_support_vector
z.add_declaration_code('''
sdcpp::ndarray KDTree_getPoint(bp::object const &bpinst, int i) {
    cv::KDTree const &inst = bp::extract<cv::KDTree const &>(bpinst);
    sdcpp::ndarray result = sdcpp::new_ndarray1d(inst.points.cols, NPY_FLOAT, 
        (void *)inst.getPoint(i));
    bp::objects::make_nurse_and_patient(result.get_obj().ptr(), bpinst.ptr());
    return result;
}
''')
z.mem_fun('getPoint').exclude()
z.add_registration_code('def( "getPoint", &KDTree_getPoint, (bp::arg("ptidx")) )')
sb.finalize_class(z)

# FileStorage
z = sb.mb.class_('FileStorage')
sb.init_class(z)
z.decls(lambda x: 'CvFileStorage' in x.decl_string).exclude()
z.operators(lambda x: '*' in x.name or 'char' in x.decl_string).exclude()
z.mem_fun('writeRaw')._transformer_creators.append(FT.input_array1d('vec', 'len'))
z.mem_fun('writeObj').exclude() # too old
for t in ('structs', 'fs'): # TODO: expose 'structs' but not 'fs'
    z.var(t).exclude()
sb.finalize_class(z)

# FileNode
z = sb.mb.class_('FileNode')
z.include_files.append("opencv_converters.hpp")
sb.init_class(z)
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
sb.finalize_class(z)


#=============================================================================
# Free functions
#=============================================================================


# free functions
for z in ('fromUtf16', 'toUtf16',
    'setNumThreads', 'getNumThreads', 'getThreadNum',
    'getTickCount', 'getTickFrequency', 'getCPUTickCount', 'checkHardwareSupport',
    'setUseOptimized', 'useOptimized',
    ):
    sb.mb.free_fun(lambda decl: z in decl.name).include()
    
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
    sb.mb.free_funs(z).include()

for t in ('min', 'max', 'sqrt', 'pow', 'exp', 'log'):
    for z in sb.mb.free_funs(t):
        if 'cv::Mat' in z.decl_string:
            z.include()

# split, merge
for t in ('split', 'merge'):
    for z in sb.mb.free_funs(t):
        if 'vector' in z.partial_decl_string:
            z.include()
        
# mixChannels
for z in sb.mb.free_funs('mixChannels'):
    if 'vector' in z.partial_decl_string:
        z.include()
        z._transformer_kwds['alias'] = 'mixChannels'
        z._transformer_creators.append(FT.input_array1d('fromTo'))

# minMaxLoc
for z in sb.mb.free_funs('minMaxLoc'):
    z.include()
    z._transformer_kwds['alias'] = 'minMaxLoc'
    for i in xrange(1,5):
        z._transformer_creators.append(FT.output_type1(i))

# checkRange
for z in sb.mb.free_funs('checkRange'):
    z.include()
    z._transformer_creators.append(FT.output_type1(2))
    z._transformer_kwds['alias'] = 'checkRange'

# kmeans
z = sb.mb.free_fun('kmeans')
z.include()
z._transformer_creators.append(FT.output_type1('centers'))

# calcCovarMatrix
for z in sb.mb.free_funs('calcCovarMatrix'):
    z.include()
    if z.arguments[0].type == D.dummy_type_t('::cv::Mat const *'):
        z._transformer_creators.append(FT.input_array1d('samples', 'nsamples'))
    z._transformer_kwds['alias'] = 'calcCovarMatrix'
        
# theRNG
z = sb.mb.free_fun('theRNG')
z.include()
z.call_policies = CP.return_value_policy(CP.reference_existing_object)

# fillConvexPoly
z = sb.mb.free_fun('fillConvexPoly')
z.include()
z._transformer_creators.append(FT.input_array1d('pts', 'npts'))

# fillPoly
for t in ('fillPoly', 'polylines'):
    z = sb.mb.free_fun(t)
    z.include()
    z._transformer_creators.append(FT.input_array2d('pts', 'npts', 'ncontours'))
    z._transformer_kwds['alias'] = t
    
# getTextSize
z = sb.mb.free_fun('getTextSize')
z.include()
z._transformer_creators.append(FT.output_type1('baseLine'))

# MemStorage -- exposed in cxtypes_h_gen

# cvCreateMemStorage
sb.mb.free_fun('cvCreateMemStorage').exclude()
sb.mb.add_declaration_code('''
cv::MemStorage createMemStorage(int block_size CV_DEFAULT(0))
{
    return cv::MemStorage(cvCreateMemStorage(block_size));
}
''')
sb.mb.add_registration_code('bp::def("createMemStorage", &::createMemStorage);')

# cvCreateChildMemStorage
sb.mb.free_fun('cvCreateChildMemStorage').exclude()
sb.mb.add_declaration_code('''
cv::MemStorage createChildMemStorage(cv::MemStorage &parent)
{
    return cv::MemStorage(cvCreateChildMemStorage((CvMemStorage *)parent));
}

''')
sb.mb.add_registration_code('bp::def("createChildMemStorage", &::createChildMemStorage, bp::with_custodian_and_ward_postcall<0,1>());')
        

# Seq
# to expose a template class Seq<T>, use expose_class_Seq('T')
sb.register_ti('int')
sb.expose_class_Seq('int')


# MatExpr
sb.mb.decls(lambda x: 'MatExpr' in x.decl_string).exclude()

sb.done()
sb.save_regs('cxcore_hpp_reg.sdd')
