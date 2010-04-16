// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "ndarray.hpp"
#include "opencv_converters.hpp"
#include "Scalar.pypp.hpp"

namespace bp = boost::python;

void register_Scalar_class(){

    { //::cv::Scalar_< double >
        typedef bp::class_< cv::Scalar_< double >, bp::bases< cv::Vec< double, 4 > > > Scalar_exposer_t;
        Scalar_exposer_t Scalar_exposer = Scalar_exposer_t( "Scalar", bp::init< >() );
        bp::scope Scalar_scope( Scalar_exposer );
        Scalar_exposer.add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::Scalar_< double > >() );
        Scalar_exposer.def( bp::init< double, double, bp::optional< double, double > >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2")=0, bp::arg("v3")=0 )) );
        Scalar_exposer.def( bp::init< CvScalar const & >(( bp::arg("s") )) );
        bp::implicitly_convertible< CvScalar const &, cv::Scalar_< double > >();
        Scalar_exposer.def( bp::init< double >(( bp::arg("v0") )) );
        bp::implicitly_convertible< double, cv::Scalar_< double > >();
        { //::cv::Scalar_< double >::all
        
            typedef cv::Scalar_< double > exported_class_t;
            typedef ::cv::Scalar_< double > ( *all_function_type )( double );
            
            Scalar_exposer.def( 
                "all"
                , all_function_type( &::cv::Scalar_< double >::all )
                , ( bp::arg("v0") ) );
        
        }
        { //::cv::Scalar_< double >::mul
        
            typedef cv::Scalar_< double > exported_class_t;
            typedef ::cv::Scalar_< double > ( exported_class_t::*mul_function_type )( ::cv::Scalar_< double > const &,double ) const;
            
            Scalar_exposer.def( 
                "mul"
                , mul_function_type( &::cv::Scalar_< double >::mul )
                , ( bp::arg("t"), bp::arg("scale")=1 ) );
        
        }
        Scalar_exposer.def( "__temp_func", &cv::Scalar_< double >::operator ::CvScalar  );
        Scalar_exposer.staticmethod( "all" );
        Scalar_exposer.def("from_ndarray", &bp::from_ndarray< cv::Scalar >, (bp::arg("arr")) );
        Scalar_exposer.staticmethod("from_ndarray");
        Scalar_exposer.add_property("ndarray", &bp::as_ndarray< cv::Scalar >);
        Scalar_exposer.def("__iadd__", &__iadd__<cv::Scalar, cv::Scalar >, bp::return_self<>() );
        Scalar_exposer.def("__isub__", &__isub__<cv::Scalar, cv::Scalar >, bp::return_self<>() );
        Scalar_exposer.def("__imul__", &__imul__<cv::Scalar, double >, bp::return_self<>() );
        Scalar_exposer.def("__add__", &__add__<cv::Scalar, cv::Scalar> );
        Scalar_exposer.def("__sub__", &__sub__<cv::Scalar, cv::Scalar> );
        Scalar_exposer.def("__ne__", &__ne__<cv::Scalar, cv::Scalar> );
        Scalar_exposer.def("__eq__", &__eq__<cv::Scalar, cv::Scalar> );
        Scalar_exposer.def("__mul__", &__mul__<cv::Scalar, double> );
        Scalar_exposer.def("__rmul__", &__rmul__<double, cv::Scalar> );
        Scalar_exposer.def("__neg__", &__neg__<cv::Scalar> );
    }

}
