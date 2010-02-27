// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "boost/python/object.hpp"
#include "boost/python/str.hpp"
#include "CvContourTree.pypp.hpp"

namespace bp = boost::python;

struct CvContourTree_wrapper : CvContourTree, bp::wrapper< CvContourTree > {

    CvContourTree_wrapper(CvContourTree const & arg )
    : CvContourTree( arg )
      , bp::wrapper< CvContourTree >(){
        // copy constructor
        
    }

    CvContourTree_wrapper()
    : CvContourTree()
      , bp::wrapper< CvContourTree >(){
        // null constructor
        
    }

    static bp::object get_block_max( ::CvContourTree const & inst ){        
        return inst.block_max? bp::str(inst.block_max): bp::object();
    }

    static bp::object get_ptr( ::CvContourTree const & inst ){        
        return inst.ptr? bp::str(inst.ptr): bp::object();
    }

};

static ::CvSeq * get_h_prev( ::CvContourTree const & inst ) { return inst.h_prev; }

static ::CvSeq * get_h_next( ::CvContourTree const & inst ) { return inst.h_next; }

static ::CvSeq * get_v_prev( ::CvContourTree const & inst ) { return inst.v_prev; }

static ::CvSeq * get_v_next( ::CvContourTree const & inst ) { return inst.v_next; }

static ::CvMemStorage * get_storage( ::CvContourTree const & inst ) { return inst.storage; }

static ::CvSeqBlock * get_free_blocks( ::CvContourTree const & inst ) { return inst.free_blocks; }

static ::CvSeqBlock * get_first( ::CvContourTree const & inst ) { return inst.first; }

void register_CvContourTree_class(){

    bp::class_< CvContourTree_wrapper >( "CvContourTree" )    
        .add_property( "this", pyplus_conv::make_addressof_inst_getter< CvContourTree >() )    
        .def_readwrite( "delta_elems", &CvContourTree::delta_elems )    
        .def_readwrite( "elem_size", &CvContourTree::elem_size )    
        .def_readwrite( "flags", &CvContourTree::flags )    
        .def_readwrite( "header_size", &CvContourTree::header_size )    
        .def_readwrite( "p1", &CvContourTree::p1 )    
        .def_readwrite( "p2", &CvContourTree::p2 )    
        .def_readwrite( "total", &CvContourTree::total )    
        .add_property( "h_prev", bp::make_function(&::get_h_prev, bp::return_internal_reference<>()) )    
        .add_property( "h_next", bp::make_function(&::get_h_next, bp::return_internal_reference<>()) )    
        .add_property( "v_prev", bp::make_function(&::get_v_prev, bp::return_internal_reference<>()) )    
        .add_property( "v_next", bp::make_function(&::get_v_next, bp::return_internal_reference<>()) )    
        .add_property( "storage", bp::make_function(&::get_storage, bp::return_internal_reference<>()) )    
        .add_property( "free_blocks", bp::make_function(&::get_free_blocks, bp::return_internal_reference<>()) )    
        .add_property( "first", bp::make_function(&::get_first, bp::return_internal_reference<>()) )    
        .add_property( "block_max", bp::make_function(&::CvContourTree_wrapper::get_block_max) )    
        .add_property( "ptr", bp::make_function(&::CvContourTree_wrapper::get_ptr) );

}
