// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "KalmanFilter.pypp.hpp"

namespace bp = boost::python;

void register_KalmanFilter_class(){

    bp::class_< cv::KalmanFilter >( "KalmanFilter", bp::init< >() )    
        .add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::KalmanFilter >() )    
        .def( bp::init< int, int, bp::optional< int > >(( bp::arg("dynamParams"), bp::arg("measureParams"), bp::arg("controlParams")=(int)(0) )) )    
        .def( 
            "correct"
            , (::cv::Mat const & ( ::cv::KalmanFilter::* )( ::cv::Mat const & ) )( &::cv::KalmanFilter::correct )
            , ( bp::arg("measurement") )
            , bp::return_self< >() )    
        .def( 
            "init"
            , (void ( ::cv::KalmanFilter::* )( int,int,int ) )( &::cv::KalmanFilter::init )
            , ( bp::arg("dynamParams"), bp::arg("measureParams"), bp::arg("controlParams")=(int)(0) ) )    
        .def( 
            "predict"
            , (::cv::Mat const & ( ::cv::KalmanFilter::* )( ::cv::Mat const & ) )( &::cv::KalmanFilter::predict )
            , ( bp::arg("control")=cv::Mat() )
            , bp::return_self< >() )    
        .def_readwrite( "controlMatrix", &cv::KalmanFilter::controlMatrix )    
        .def_readwrite( "errorCovPost", &cv::KalmanFilter::errorCovPost )    
        .def_readwrite( "errorCovPre", &cv::KalmanFilter::errorCovPre )    
        .def_readwrite( "gain", &cv::KalmanFilter::gain )    
        .def_readwrite( "measurementMatrix", &cv::KalmanFilter::measurementMatrix )    
        .def_readwrite( "measurementNoiseCov", &cv::KalmanFilter::measurementNoiseCov )    
        .def_readwrite( "processNoiseCov", &cv::KalmanFilter::processNoiseCov )    
        .def_readwrite( "statePost", &cv::KalmanFilter::statePost )    
        .def_readwrite( "statePre", &cv::KalmanFilter::statePre )    
        .def_readwrite( "temp1", &cv::KalmanFilter::temp1 )    
        .def_readwrite( "temp2", &cv::KalmanFilter::temp2 )    
        .def_readwrite( "temp3", &cv::KalmanFilter::temp3 )    
        .def_readwrite( "temp4", &cv::KalmanFilter::temp4 )    
        .def_readwrite( "temp5", &cv::KalmanFilter::temp5 )    
        .def_readwrite( "transitionMatrix", &cv::KalmanFilter::transitionMatrix );

}
