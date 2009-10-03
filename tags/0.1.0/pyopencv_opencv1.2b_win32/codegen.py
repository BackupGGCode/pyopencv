
import os
from pyplusplus import module_builder
from pyplusplus import messages

#Creating an instance of class that will help you to expose your declarations
mb = module_builder.module_builder_t( 
    [
        "pyopencvext.hpp", 
        # r"M:/programming/mypackages/pyopencv/workspace_svn/pyopencv_opencv1.2b_win32/pyopencvext.hpp"
    ],
    gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe", 
    working_directory=r"M:/programming/mypackages/pyopencv/workspace_svn/pyopencv_opencv1.2b_win32", 
    include_paths=[
        r"M:/programming/mypackages/pyopencv/workspace_svn/pyopencv_opencv1.2b_win32",
        r"M:/programming/packages/opencv/build/svn/include",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include\c++\mingw32",
        r"M:\programming\builders\MinGW\gcc\gcc-4.4.0-mingw\lib\gcc\mingw32\4.4.0\include",
    ],
    define_symbols=[] )


#Well, don't you want to see what is going on?
# mb.print_declarations() -- too many declarations

# Disable every declarations first
# mb.decls().exclude()


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
    

    
# -----------------------------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------------------------
# disable some warnings
mb.decls().disable_warnings(messages.W1027, messages.W1025)

# exlude every class first
mb.classes().exclude()

# expose every OpenCV's C structure and class but none of its members
for z in mb.classes(lambda z: z.decl_string.startswith('::Cv') or z.decl_string.startswith('::_Ipl')):
    z.include()
    z.decls().exclude()
    
# exclude stupid CvMat... aliases
mb.classes(lambda z: z.decl_string.startswith('::CvMat') and not z.name.startswith('CvMat')).exclude()
    
# cannot expose unions
mb.class_('Cv32suf').exclude()
mb.class_('Cv64suf').exclude()

# expose every OpenCV's C++ class but none of its members
for z in mb.classes(lambda z: z.decl_string.startswith('::cv')):
    z.include()
    z.decls().exclude()
    
# exclude every Ptr class
mb.classes(lambda z: z.decl_string.startswith('::cv::Ptr')).exclude()

# exclude every MatExpr class
mb.classes(lambda z: z.decl_string.startswith('::cv::MatExpr')).exclude()

# expose every OpenCV's C++ free function
mb.free_functions(lambda z: z.decl_string.startswith('::cv')).include()

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

for z in ('CvScalar', 'CvPoint', 'CvSize', 'CvRect', 'CvBox', 'CvSlice'):
    mb.decls(lambda decl: decl.name.startswith(z)).include()


# -----------------------------------------------------------------------------------------------
# cxcore.hpp
# -----------------------------------------------------------------------------------------------
cv = mb.namespace('cv') # namespace cv

cv.class_('Exception').include()

# for z in ('Optimized', 'NumThreads', 'ThreadNum', 'getTick'):
    # cv.decls(lambda decl: z in decl.name).include()

for z in ('DataDepth', 'Vec', 'Complex', 'Point', 'Size', 'Rect', 'RotatedRect', 
    'Scalar', 'Range', 'DataType'):
    cv.decls(lambda decl: decl.name.startswith(z)).include()

# class Mat    
mat = cv.class_('Mat')
mat.include()
for z in ('refcount', 'datastart', 'dataend'):
    mat.var(z).exclude()
# TODO: expose the 'data' member as read-write buffer
mat.var('data').exclude()
# expose_addressof_member(mat, 'data')    
mat.decls('ptr').exclude()

#Creating code creator. After this step you should not modify/customize declarations.
mb.build_code_creator( module_name='pyopencvext' )

#Writing code to file.
mb.split_module( 'code' )
