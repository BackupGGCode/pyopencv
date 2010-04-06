
import os
from pyplusplus import module_builder
from pyplusplus.module_builder import call_policies

opencv_path = r"M:/programming/packages/OpenCV/build/1.1a"

cx_include_path = opencv_path + r"/cxcore/include"
cv_include_path = opencv_path + r"/cv/include"
hg_include_path = opencv_path + r"/otherlibs/highgui"
ml_include_path = opencv_path + r"/ml/include"


#Creating an instance of class that will help you to expose your declarations
mb = module_builder.module_builder_t( [cx_include_path+r"/cxcore.h"]
# [cx_include_path+r"/cxcore.h", cv_include_path+r"/cv.h", hg_include_path+r"/highgui.h", ml_include_path+r"/ml.h"]
                                      , gccxml_path=r"M:/utils/gccxml/bin/gccxml.exe" 
                                      , working_directory=r"."
                                      , include_paths=[cx_include_path, cv_include_path, hg_include_path, ml_include_path]
                                      , define_symbols=[] )


#Well, don't you want to see what is going on?
# mb.print_declarations()

# Disable some declarations
decl_list = [
    'cvSetIPLAllocators', 
]
for x in decl_list:
    mb.decl(x).exclude()

# Disable some classes
class_list = [
    'CvType', 'CvImage', 'Cv64suf', 'Cv32suf',
]
for x in class_list:
    mb.class_(x).exclude()

# Convert member variables which are pointers into addresses
mem_var_ptr_dict = {
    '_IplImage': ['imageId', 'imageData', 'imageDataOrigin'],
    'CvModuleInfo': ['name', 'version'],
}
for class_name, attrs in mem_var_ptr_dict.items():
    klass = mb.class_(class_name)
    for attr in attrs:
        item = klass.var(attr)
        item.rename('addressof_'+attr)
        item.expose_address = True

# Rename IplImage
mb.class_('_IplImage').rename('IplImage')

# Disable cvRelease*
mb.decls( lambda decl: decl.name.startswith('cvRelease') ).exclude()

# Function policies for CvMat 2d slicing functions
# cvmat_slice2d_list = [
    # 'cvGetCol', 'cvGetRow', 'cvGetDiag',
# ]
# for x in cvmat_slice2d_list:
    # mb.free_function(x).call_policies = call_policies.return_internal_reference(2, call_policies.with_custodian_and_ward_postcall(2,1))

# Fix cvGetCol
mb.add_declaration_code("""
::CvMat * sdGetCol( ::CvArr const *, int )
{
    return 0;
}
""", tail=True)
   
#Creating code creator. After this step you should not modify/customize declarations.
mb.build_code_creator( module_name='cxcore' )

#Writing code to file.
mb.write_module( '../src/opencv.cpp' )
    

