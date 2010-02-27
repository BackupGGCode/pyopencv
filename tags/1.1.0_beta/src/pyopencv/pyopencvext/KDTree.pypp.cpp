// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "KDTree.pypp.hpp"

namespace bp = boost::python;

void register_KDTree_class(){

    bp::class_< cv::KDTree >( "KDTree" )    
        .add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::KDTree >() );

}
