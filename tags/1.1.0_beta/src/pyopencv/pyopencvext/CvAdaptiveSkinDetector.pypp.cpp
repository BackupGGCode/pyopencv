// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "CvAdaptiveSkinDetector.pypp.hpp"

namespace bp = boost::python;

void register_CvAdaptiveSkinDetector_class(){

    bp::class_< CvAdaptiveSkinDetector >( "CvAdaptiveSkinDetector", bp::no_init )    
        .add_property( "this", pyplus_conv::make_addressof_inst_getter< CvAdaptiveSkinDetector >() );

}
