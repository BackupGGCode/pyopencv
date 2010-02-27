// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "ndarray.hpp"
#include "Vec4d.pypp.hpp"

namespace bp = boost::python;

void register_Vec4d_class(){

    { //::cv::Vec< double, 4 >
        typedef bp::class_< cv::Vec< double, 4 > > Vec4d_exposer_t;
        Vec4d_exposer_t Vec4d_exposer = Vec4d_exposer_t( "Vec4d", bp::init< >() );
        bp::scope Vec4d_scope( Vec4d_exposer );
        Vec4d_exposer.add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::Vec< double, 4 > >() );
        bp::scope().attr("depth") = (int)cv::Vec<double, 4>::depth;
        bp::scope().attr("channels") = (int)cv::Vec<double, 4>::channels;
        bp::scope().attr("type") = (int)cv::Vec<double, 4>::type;
        Vec4d_exposer.def( bp::init< double >(( bp::arg("v0") )) );
        bp::implicitly_convertible< double, cv::Vec< double, 4 > >();
        Vec4d_exposer.def( bp::init< double, double >(( bp::arg("v0"), bp::arg("v1") )) );
        Vec4d_exposer.def( bp::init< double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4"), bp::arg("v5") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4"), bp::arg("v5"), bp::arg("v6") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4"), bp::arg("v5"), bp::arg("v6"), bp::arg("v7") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4"), bp::arg("v5"), bp::arg("v6"), bp::arg("v7"), bp::arg("v8") )) );
        Vec4d_exposer.def( bp::init< double, double, double, double, double, double, double, double, double, double >(( bp::arg("v0"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3"), bp::arg("v4"), bp::arg("v5"), bp::arg("v6"), bp::arg("v7"), bp::arg("v8"), bp::arg("v9") )) );
        Vec4d_exposer.def( bp::init< cv::Vec< double, 4 > const & >(( bp::arg("v") )) );
        { //::cv::Vec< double, 4 >::all
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef ::cv::Vec< double, 4 > ( *all_function_type )( double );
            
            Vec4d_exposer.def( 
                "all"
                , all_function_type( &::cv::Vec< double, 4 >::all )
                , ( bp::arg("alpha") ) );
        
        }
        { //::cv::Vec< double, 4 >::cross
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef ::cv::Vec< double, 4 > ( exported_class_t::*cross_function_type )( ::cv::Vec< double, 4 > const & ) const;
            
            Vec4d_exposer.def( 
                "cross"
                , cross_function_type( &::cv::Vec< double, 4 >::cross )
                , ( bp::arg("v") ) );
        
        }
        { //::cv::Vec< double, 4 >::ddot
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef double ( exported_class_t::*ddot_function_type )( ::cv::Vec< double, 4 > const & ) const;
            
            Vec4d_exposer.def( 
                "ddot"
                , ddot_function_type( &::cv::Vec< double, 4 >::ddot )
                , ( bp::arg("v") ) );
        
        }
        { //::cv::Vec< double, 4 >::dot
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef double ( exported_class_t::*dot_function_type )( ::cv::Vec< double, 4 > const & ) const;
            
            Vec4d_exposer.def( 
                "dot"
                , dot_function_type( &::cv::Vec< double, 4 >::dot )
                , ( bp::arg("v") ) );
        
        }
        { //::cv::Vec< double, 4 >::operator[]
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef double ( exported_class_t::*__getitem___function_type )( int ) const;
            
            Vec4d_exposer.def( 
                "__getitem__"
                , __getitem___function_type( &::cv::Vec< double, 4 >::operator[] )
                , ( bp::arg("i") ) );
        
        }
        { //::cv::Vec< double, 4 >::operator[]
        
            typedef cv::Vec< double, 4 > exported_class_t;
            typedef double & ( exported_class_t::*__getitem___function_type )( int ) ;
            
            Vec4d_exposer.def( 
                "__getitem__"
                , __getitem___function_type( &::cv::Vec< double, 4 >::operator[] )
                , ( bp::arg("i") )
                , bp::return_value_policy< bp::copy_non_const_reference >() );
        
        }
        Vec4d_exposer.staticmethod( "all" );
        Vec4d_exposer.def("from_ndarray", &bp::from_ndarray< cv::Vec4d >, (bp::arg("arr")) );
        Vec4d_exposer.staticmethod("from_ndarray");
        Vec4d_exposer.add_property("ndarray", &bp::as_ndarray< cv::Vec4d >);
    }

}
