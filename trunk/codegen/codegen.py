
import os
from pygccxml import declarations
from pyplusplus import module_builder, messages
from pyplusplus import function_transformers as FT
from pyplusplus.module_builder import call_policies as CP

#Creating an instance of class that will help you to expose your declarations
mb = module_builder.module_builder_t( 
    [
        "cxcore.h", 
        # "cxcore.hpp", 
        # "cv.h", 
        # "cv.hpp", 
        # "cvaux.h", 
        # "cvaux.hpp", 
        # "ml.h", 
        # "highgui.h", 
        # "highgui.hpp",
        # r"M:/programming/mypackages/pyopencv/workspace_svn/pyopencv_opencv1.2b_win32/pyopencvext.hpp"
    ],
    gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe", 
    working_directory=r"M:/programming/mypackages/pyopencv/svn_workplace/trunk/codegen", 
    include_paths=[
        r"M:/programming/mypackages/pyopencv/svn_workplace/trunk/codegen/opencv2_include",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
    ],
    define_symbols=[] )


#Well, don't you want to see what is going on?
# mb.print_declarations() -- too many declarations

# Disable every declarations first
mb.decls().exclude()


def expose_addressof_member(klass, member_name, exclude_member=True):
    klass.include_files.append( "boost/python/long.hpp" )
    if exclude_member:
        klass.var(member_name).exclude()
    klass.add_declaration_code('''
    boost::python::long_ get_MEMBER_NAME_addr( CLASS_TYPE const & inst ){
        return boost::python::long_((int)&inst.MEMBER_NAME);
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('def( "get_MEMBER_NAME_addr", &::get_MEMBER_NAME_addr )'.replace("MEMBER_NAME", member_name))
    # TODO: finish the wrapping with ctypes code
    
def expose_member_as_str(klass, member_name):
    klass.include_files.append( "boost/python/object.hpp" )
    klass.include_files.append( "boost/python/str.hpp" )
    klass.var(member_name).exclude()
    klass.add_wrapper_code('''
    static bp::object get_MEMBER_NAME( CLASS_TYPE const & inst ){        
        return inst.MEMBER_NAME? bp::str(inst.MEMBER_NAME): bp::object();
    }
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    klass.add_registration_code('''
    add_property( "MEMBER_NAME", bp::make_function(&CLASS_TYPE_wrapper::get_MEMBER_NAME) )
    '''.replace("MEMBER_NAME", member_name).replace("CLASS_TYPE", klass.decl_string))
    

def remove_ptr( type_ ):
    if declarations.is_pointer( type_ ):
        return declarations.remove_pointer( type_ )
    else:
        raise TypeError( 'Type should be a pointer, got %s.' % type_ )


# -----------------------------------------------------------------------------------------------
# Function transfomers
# -----------------------------------------------------------------------------------------------

# input_double_pointee_t
class input_double_pointee_t(FT.transformer_t):
    """Handles a double pointee input.
    
    Convert by dereferencing: do_smth(your_type **v) -> do_smth(your_type v)

    Right now compiler should be able to use implicit conversion
    """

    def __init__(self, function, arg_ref):
        FT.transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )
        if not declarations.is_pointer( self.arg.type ):
            raise ValueError( '%s\nin order to use "input_double_pointee_t" transformation, argument %s type must be a pointer or a array (got %s).' ) \
                  % ( function, self.arg_ref.name, arg.type)

    def __str__(self):
        return "input_double_pointee(%s)" % self.arg.name

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        tmp_type = remove_ptr( self.arg.type )
        w_arg.type = remove_ptr( tmp_type )
        if not declarations.is_convertible( w_arg.type, self.arg.type ):
            controller.add_pre_call_code("%s tmp = reinterpret_cast< %s >(& %s);" % ( tmp_type, tmp_type, w_arg.name ))
            casting_code = 'reinterpret_cast< %s >( & tmp )' % self.arg.type
            controller.modify_arg_expression(self.arg_index, casting_code)

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return []

def input_double_pointee( *args, **keywd ):
    def creator( function ):
        return input_double_pointee_t( function, *args, **keywd )
    return creator

# input_smart_pointee_t
class input_smart_pointee_t(FT.transformer_t):
    """Handles a pointee input.
    
    Convert by dereferencing: do_smth(your_type *v) -> do_smth(object v2)
    where v2 is either of type NoneType or type 'your_type'. 
    If v2 is None, v is NULL.  Otherwise, v is the pointer to v2.
    """

    def __init__(self, function, arg_ref):
        FT.transformer.transformer_t.__init__( self, function )
        self.arg = self.get_argument( arg_ref )
        self.arg_index = self.function.arguments.index( self.arg )

    def __str__(self):
        return "input_smart_pointee(%s)" % self.arg.name

    def required_headers( self ):
        """Returns list of header files that transformer generated code depends on."""
        return [ "boost/python/object.hpp" ]

    def __configure_sealed( self, controller ):
        w_arg = controller.find_wrapper_arg( self.arg.name )
        data_type = remove_ptr( self.arg.type )
        w_arg.type = declarations.dummy_type_t( "boost::python::object" )
        casting_code = "(%s.ptr() != bp::Py_None)? 1: &bp::extract<%s>(%s)" % (w_arg.name, data_type, w_arg.name)
        controller.modify_arg_expression(self.arg_index, casting_code)

    def __configure_v_mem_fun_default( self, controller ):
        self.__configure_sealed( controller )

    def configure_mem_fun( self, controller ):
        self.__configure_sealed( controller )

    def configure_free_fun(self, controller ):
        self.__configure_sealed( controller )

    def configure_virtual_mem_fun( self, controller ):
        self.__configure_v_mem_fun_default( controller.default_controller )


def input_smart_pointee( *args, **keywd ):
    def creator( function ):
        return input_smart_pointee_t( function, *args, **keywd )
    return creator

    
    
#=============================================================================
# Initialization
#=============================================================================

# disable some warnings
mb.decls().disable_warnings(messages.W1027, messages.W1025)

# expose 'this'
mb.classes().expose_this = True


#=============================================================================
# CxCore
#=============================================================================



# -----------------------------------------------------------------------------------------------
# CxCore/Basic Structures
# -----------------------------------------------------------------------------------------------

for z in ('CvScalar', 'cvRealScalar', 
    'CvPoint', 'cvPoint', 
    'CvSize', 'cvSize', 
    'CvTermCriteria', 'cvTermCriteria', 'cvCheckTermCriteria'):
    mb.decls(lambda decl: decl.name.startswith(z)).include()
for z in ('cvScalar', 'cvScalarAll', 
    'cvRect'):
    mb.decl(z).include()
mb.class_('CvRect').include()
    
# CvMat
cvmat = mb.class_('CvMat')
cvmat.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmat.var(z).exclude()
# deal with 'data'
cvmat.include_files.append( "boost/python/object.hpp" )
cvmat.include_files.append( "boost/python/str.hpp" )
cvmat.add_wrapper_code('''
static bp::object get_data( CvMat const & inst ){        
    return bp::str((const char *)inst.data.ptr, (inst.step? inst.step: CV_ELEM_SIZE(inst.type)*inst.cols)*inst.rows);
}
''')
cvmat.add_registration_code('''
add_property( "data", bp::make_function(&CvMat_wrapper::get_data) )
''')

# CvMatND
cvmatnd = mb.class_('CvMatND')
cvmatnd.include()
for z in ('ptr', 's', 'i', 'fl', 'db', 'data'):
    cvmatnd.var(z).exclude()
# deal with 'data'
cvmatnd.include_files.append( "boost/python/object.hpp" )
cvmatnd.include_files.append( "boost/python/str.hpp" )
cvmatnd.add_wrapper_code('''
static bp::object get_data( CvMatND const & inst ){        
    return bp::str((const char *)inst.data.ptr, inst.dim[0].step*inst.dim[0].size);
}
''')
cvmatnd.add_registration_code('''
add_property( "data", bp::make_function(&CvMatND_wrapper::get_data) )
''')

# CvSparseMat
cvsparsemat = mb.class_('CvSparseMat')
cvsparsemat.include()
for z in ('heap', 'hashtable'): # TODO: fix
    cvsparsemat.var(z).exclude()
    
# IplImage
iplimage = mb.class_('_IplImage')
iplimage.rename('IplImage')
iplimage.include()
for z in ('imageId', 'imageData', 'roi', 'imageDataOrigin', 'tileInfo', 'maskROI'): # don't need these attributes
    iplimage.var(z).exclude()
# deal with 'imageData'
iplimage.include_files.append( "boost/python/object.hpp" )
iplimage.include_files.append( "boost/python/str.hpp" )
iplimage.add_wrapper_code('''
static bp::object get_data( _IplImage const & inst ){        
    return bp::str(inst.imageData, inst.imageSize);
}
''')
iplimage.add_registration_code('''
add_property( "data", bp::make_function(&_IplImage_wrapper::get_data) )
''')

# CvArr
mb.decl('CvArr').include()

def add_underscore(decl):
    decl.rename('_'+decl.name)
    decl.include()

# -----------------------------------------------------------------------------------------------
# CxCore/Operations on Arrays
# -----------------------------------------------------------------------------------------------

# return pointee value
for z in ('IplImage', 'CvMat', 'CvMatND'):
    mb.free_functions( return_type='::'+z+' *' ).call_policies \
        = CP.return_value_policy( CP.return_pointee_value )


# convert every cvRelease...() function into private        
mb.decls(lambda decl: decl.name.startswith('cvCreateImage')).include()

# cvReleaseImage... functions
for z in mb.free_funs(lambda decl: decl.name.startswith('cvReleaseImage')):
    add_underscore(z)
    z.add_transformation(input_double_pointee(0))

# cvReleaseData
z = mb.free_fun('cvReleaseData')
add_underscore(z)
z.add_transformation(input_smart_pointee(0))

# -----------------------------------------------------------------------------------------------
# Final tasks
# -----------------------------------------------------------------------------------------------

for z in ('hdr_refcount', 'refcount'): # too low-level
    mb.decls(z).exclude() 

# mb.free_function( return_type='IplImage *' ).call_policies \
    # = call_policies.return_value_policy( call_policies.return_pointee_value )


for z in ('IPL_', 'CV_'):
    try:
        mb.decls(lambda decl: decl.name.startswith(z)).include()
    except RuntimeError:
        pass













# exlude every class first
# mb.classes().exclude()

# expose every OpenCV's C structure and class but none of its members
# for z in mb.classes(lambda z: z.decl_string.startswith('::Cv') or z.decl_string.startswith('::_Ipl')):
    # z.include()
    # z.decls().exclude()
    
# exclude stupid CvMat... aliases
# mb.classes(lambda z: z.decl_string.startswith('::CvMat') and not z.name.startswith('CvMat')).exclude()
    
# cannot expose unions
# mb.class_('Cv32suf').exclude()
# mb.class_('Cv64suf').exclude()

# expose every OpenCV's C++ class but none of its members
# for z in mb.classes(lambda z: z.decl_string.startswith('::cv')):
    # z.include()
    # z.decls().exclude()
    
# exclude every Ptr class
# mb.classes(lambda z: z.decl_string.startswith('::cv::Ptr')).exclude()

# exclude every MatExpr class
# mb.classes(lambda z: z.decl_string.startswith('::cv::MatExpr')).exclude()

# expose every OpenCV's C++ free function
# mb.free_functions(lambda z: z.decl_string.startswith('::cv')).include()

# -----------------------------------------------------------------------------------------------
# cxtypes.h
# -----------------------------------------------------------------------------------------------

# CvTypeInfo
# cvtypeinfo = mb.class_('CvTypeInfo')
# expose_member_as_str(cvtypeinfo, 'type_name')
# for z in ('is_instance', 'release', 'read', 'write', 'clone'):
    # expose_addressof_member(cvtypeinfo, z)
    
# IplImage
# iplimage = mb.class_('_IplImage')
# for z in ('imageId', 'imageData', 'imageDataOrigin'):
    # iplimage.var(z).expose_address = True
    
# CvMat
# cvmat = mb.class_('CvMat')
# cvmat.include()
# cvmat.var('refcount').expose_address = True
# for z in ('ptr', 's', 'i', 'fl', 'db'):
    # cvmat.var(z).exclude()
# expose_addressof_member(cvmat, 'data')

# CvAttrList


# -----------------------------------------------------------------------------------------------

# for z in ('_IplImage', 'CvAttrList', 'CvFileNode', 'CvMatND', '_IplConvKernelFP', 
    # 'CvModuleInfo', 'CvChain', 'CvHistogram', 'CvSeqReader', 'CvContour',
    # 'CvString', 'CvSet', 'CvGraph', 'CvSeqWriter', 'CvSeq', 'CvSeqBlock', 'CvGraphEdge',
    # '_IplConvKernel', 'CvPluginFuncInfo', 'CvLineIterator', 'CvSparseMat', 'CvString',
    # '_IplROI', ):
    # mb.class_(z).exclude()
    
    
# cv = mb.namespace('cv')
# cv.decls().exclude()

# cv.decls(lambda decl: 'Optimized' in decl.name).include()

# for z in ('CvScalar', 'CvPoint', 'CvSize', 'CvRect', 'CvBox', 'CvSlice'):
    # mb.decls(lambda decl: decl.name.startswith(z)).include()


# -----------------------------------------------------------------------------------------------
# cxcore.hpp
# -----------------------------------------------------------------------------------------------
# cv = mb.namespace('cv') # namespace cv

# cv.class_('Exception').include()

# for z in ('Optimized', 'NumThreads', 'ThreadNum', 'getTick'):
    # cv.decls(lambda decl: z in decl.name).include()

# for z in ('DataDepth', 'Vec', 'Complex', 'Point', 'Size', 'Rect', 'RotatedRect', 
    # 'Scalar', 'Range', 'DataType'):
    # cv.decls(lambda decl: decl.name.startswith(z)).include()

# class Mat    
# mat = cv.class_('Mat')
# mat.include()
# for z in ('refcount', 'datastart', 'dataend'):
    # mat.var(z).exclude()
# TODO: expose the 'data' member as read-write buffer
# mat.var('data').exclude()
# expose_addressof_member(mat, 'data')    
# mat.decls('ptr').exclude()

#Creating code creator. After this step you should not modify/customize declarations.
mb.build_code_creator( module_name='pyopencvext' )

#Writing code to file.
mb.split_module( 'code' )
