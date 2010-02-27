// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "Complexd.pypp.hpp"

namespace bp = boost::python;

void register_Complexd_class(){

    { //::cv::Complex< double >
        typedef bp::class_< cv::Complex< double > > Complexd_exposer_t;
        Complexd_exposer_t Complexd_exposer = Complexd_exposer_t( "Complexd", bp::init< >() );
        bp::scope Complexd_scope( Complexd_exposer );
        Complexd_exposer.add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::Complex< double > >() );
        Complexd_exposer.def( bp::init< double, bp::optional< double > >(( bp::arg("_re"), bp::arg("_im")=0 )) );
        bp::implicitly_convertible< double, cv::Complex< double > >();
        { //::cv::Complex< double >::conj
        
            typedef cv::Complex< double > exported_class_t;
            typedef ::cv::Complex< double > ( exported_class_t::*conj_function_type )(  ) const;
            
            Complexd_exposer.def( 
                "conj"
                , conj_function_type( &::cv::Complex< double >::conj ) );
        
        }
        Complexd_exposer.def_readwrite( "im", &cv::Complex< double >::im );
        Complexd_exposer.def_readwrite( "re", &cv::Complex< double >::re );
    }

}
