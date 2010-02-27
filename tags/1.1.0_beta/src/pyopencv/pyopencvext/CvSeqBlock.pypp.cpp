// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "CvSeqBlock.pypp.hpp"

namespace bp = boost::python;

struct CvSeqBlock_wrapper : CvSeqBlock, bp::wrapper< CvSeqBlock > {

    CvSeqBlock_wrapper(CvSeqBlock const & arg )
    : CvSeqBlock( arg )
      , bp::wrapper< CvSeqBlock >(){
        // copy constructor
        
    }

    CvSeqBlock_wrapper()
    : CvSeqBlock()
      , bp::wrapper< CvSeqBlock >(){
        // null constructor
        
    }

    static bp::object get_data( ::CvSeqBlock const & inst ){        
        return inst.data? bp::str(inst.data): bp::object();
    }

};

static ::CvSeqBlock * get_prev( ::CvSeqBlock const & inst ) { return inst.prev; }

static ::CvSeqBlock * get_next( ::CvSeqBlock const & inst ) { return inst.next; }

void register_CvSeqBlock_class(){

    bp::class_< CvSeqBlock_wrapper >( "CvSeqBlock" )    
        .add_property( "this", pyplus_conv::make_addressof_inst_getter< CvSeqBlock >() )    
        .def_readwrite( "count", &CvSeqBlock::count )    
        .def_readwrite( "start_index", &CvSeqBlock::start_index )    
        .add_property( "prev", bp::make_function(&::get_prev, bp::return_internal_reference<>()) )    
        .add_property( "next", bp::make_function(&::get_next, bp::return_internal_reference<>()) )    
        .add_property( "data", bp::make_function(&::CvSeqBlock_wrapper::get_data) );

}
