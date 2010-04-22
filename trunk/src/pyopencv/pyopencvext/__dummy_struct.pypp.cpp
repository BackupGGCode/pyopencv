// This file has been generated by Py++.

#include "boost/python.hpp"
#include "__ctypes_integration.pypp.hpp"
#include "opencv_headers.hpp"
#include "opencv_converters.hpp"
#include "sequence.hpp"
#include "ndarray.hpp"
#include "__dummy_struct.pypp.hpp"

namespace bp = boost::python;

void register___dummy_struct_class(){

    { //::cv::dummy_struct
        typedef bp::class_< cv::dummy_struct > __dummy_struct_exposer_t;
        __dummy_struct_exposer_t __dummy_struct_exposer = __dummy_struct_exposer_t( "__dummy_struct" );
        bp::scope __dummy_struct_scope( __dummy_struct_exposer );
        __dummy_struct_exposer.add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::dummy_struct >() );
        bp::class_< cv::dummy_struct::dummy_struct2 >( "dummy_struct2" )    
            .add_property( "this", pyplus_conv::make_addressof_inst_getter< cv::dummy_struct::dummy_struct2 >() );
        __dummy_struct_exposer.setattr("v0", 0);
    }
    {
        
        sdcpp::register_sdobject<sdcpp::sequence>();
        sdcpp::register_sdobject<sdcpp::ndarray>();
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned short, 2 >, ::CvScalar >, (bp::arg("inst_Vec2w")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< double, 4 >, ::CvScalar >, (bp::arg("inst_Vec4d")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned short, 4 >, ::CvScalar >, (bp::arg("inst_Vec4w")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< double, 2 >, ::CvScalar >, (bp::arg("inst_Vec2d")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned short, 3 >, ::CvScalar >, (bp::arg("inst_Vec3w")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< int, 2 >, ::CvScalar >, (bp::arg("inst_Vec2i")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< int, 3 >, ::CvScalar >, (bp::arg("inst_Vec3i")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< double, 3 >, ::CvScalar >, (bp::arg("inst_Vec3d")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< int, 4 >, ::CvScalar >, (bp::arg("inst_Vec4i")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< double, 6 >, ::CvScalar >, (bp::arg("inst_Vec6d")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< short, 3 >, ::CvScalar >, (bp::arg("inst_Vec3s")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< short, 2 >, ::CvScalar >, (bp::arg("inst_Vec2s")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< short, 4 >, ::CvScalar >, (bp::arg("inst_Vec4s")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned char, 2 >, ::CvScalar >, (bp::arg("inst_Vec2b")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< float, 3 >, ::CvScalar >, (bp::arg("inst_Vec3f")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned char, 4 >, ::CvScalar >, (bp::arg("inst_Vec4b")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< float, 6 >, ::CvScalar >, (bp::arg("inst_Vec6f")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< float, 2 >, ::CvScalar >, (bp::arg("inst_Vec2f")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< float, 4 >, ::CvScalar >, (bp::arg("inst_Vec4f")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Vec< unsigned char, 3 >, ::CvScalar >, (bp::arg("inst_Vec3b")));
        bp::def("asVec2s", &::normal_cast< ::cv::Vec<unsigned char, 2>, ::cv::Vec<short, 2> >, (bp::arg("inst_Vec2b")));
        bp::def("asVec2w", &::normal_cast< ::cv::Vec<unsigned char, 2>, ::cv::Vec<unsigned short, 2> >, (bp::arg("inst_Vec2b")));
        bp::def("asVec2i", &::normal_cast< ::cv::Vec<unsigned char, 2>, ::cv::Vec<int, 2> >, (bp::arg("inst_Vec2b")));
        bp::def("asVec2f", &::normal_cast< ::cv::Vec<unsigned char, 2>, ::cv::Vec<float, 2> >, (bp::arg("inst_Vec2b")));
        bp::def("asVec2d", &::normal_cast< ::cv::Vec<unsigned char, 2>, ::cv::Vec<double, 2> >, (bp::arg("inst_Vec2b")));
        bp::def("asVec2b", &::normal_cast< ::cv::Vec<short, 2>, ::cv::Vec<unsigned char, 2> >, (bp::arg("inst_Vec2s")));
        bp::def("asVec2w", &::normal_cast< ::cv::Vec<short, 2>, ::cv::Vec<unsigned short, 2> >, (bp::arg("inst_Vec2s")));
        bp::def("asVec2i", &::normal_cast< ::cv::Vec<short, 2>, ::cv::Vec<int, 2> >, (bp::arg("inst_Vec2s")));
        bp::def("asVec2f", &::normal_cast< ::cv::Vec<short, 2>, ::cv::Vec<float, 2> >, (bp::arg("inst_Vec2s")));
        bp::def("asVec2d", &::normal_cast< ::cv::Vec<short, 2>, ::cv::Vec<double, 2> >, (bp::arg("inst_Vec2s")));
        bp::def("asVec2b", &::normal_cast< ::cv::Vec<unsigned short, 2>, ::cv::Vec<unsigned char, 2> >, (bp::arg("inst_Vec2w")));
        bp::def("asVec2s", &::normal_cast< ::cv::Vec<unsigned short, 2>, ::cv::Vec<short, 2> >, (bp::arg("inst_Vec2w")));
        bp::def("asVec2i", &::normal_cast< ::cv::Vec<unsigned short, 2>, ::cv::Vec<int, 2> >, (bp::arg("inst_Vec2w")));
        bp::def("asVec2f", &::normal_cast< ::cv::Vec<unsigned short, 2>, ::cv::Vec<float, 2> >, (bp::arg("inst_Vec2w")));
        bp::def("asVec2d", &::normal_cast< ::cv::Vec<unsigned short, 2>, ::cv::Vec<double, 2> >, (bp::arg("inst_Vec2w")));
        bp::def("asVec2b", &::normal_cast< ::cv::Vec<int, 2>, ::cv::Vec<unsigned char, 2> >, (bp::arg("inst_Vec2i")));
        bp::def("asVec2s", &::normal_cast< ::cv::Vec<int, 2>, ::cv::Vec<short, 2> >, (bp::arg("inst_Vec2i")));
        bp::def("asVec2w", &::normal_cast< ::cv::Vec<int, 2>, ::cv::Vec<unsigned short, 2> >, (bp::arg("inst_Vec2i")));
        bp::def("asVec2f", &::normal_cast< ::cv::Vec<int, 2>, ::cv::Vec<float, 2> >, (bp::arg("inst_Vec2i")));
        bp::def("asVec2d", &::normal_cast< ::cv::Vec<int, 2>, ::cv::Vec<double, 2> >, (bp::arg("inst_Vec2i")));
        bp::def("asVec2b", &::normal_cast< ::cv::Vec<float, 2>, ::cv::Vec<unsigned char, 2> >, (bp::arg("inst_Vec2f")));
        bp::def("asVec2s", &::normal_cast< ::cv::Vec<float, 2>, ::cv::Vec<short, 2> >, (bp::arg("inst_Vec2f")));
        bp::def("asVec2w", &::normal_cast< ::cv::Vec<float, 2>, ::cv::Vec<unsigned short, 2> >, (bp::arg("inst_Vec2f")));
        bp::def("asVec2i", &::normal_cast< ::cv::Vec<float, 2>, ::cv::Vec<int, 2> >, (bp::arg("inst_Vec2f")));
        bp::def("asVec2d", &::normal_cast< ::cv::Vec<float, 2>, ::cv::Vec<double, 2> >, (bp::arg("inst_Vec2f")));
        bp::def("asVec2b", &::normal_cast< ::cv::Vec<double, 2>, ::cv::Vec<unsigned char, 2> >, (bp::arg("inst_Vec2d")));
        bp::def("asVec2s", &::normal_cast< ::cv::Vec<double, 2>, ::cv::Vec<short, 2> >, (bp::arg("inst_Vec2d")));
        bp::def("asVec2w", &::normal_cast< ::cv::Vec<double, 2>, ::cv::Vec<unsigned short, 2> >, (bp::arg("inst_Vec2d")));
        bp::def("asVec2i", &::normal_cast< ::cv::Vec<double, 2>, ::cv::Vec<int, 2> >, (bp::arg("inst_Vec2d")));
        bp::def("asVec2f", &::normal_cast< ::cv::Vec<double, 2>, ::cv::Vec<float, 2> >, (bp::arg("inst_Vec2d")));
        bp::def("asVec3s", &::normal_cast< ::cv::Vec<unsigned char, 3>, ::cv::Vec<short, 3> >, (bp::arg("inst_Vec3b")));
        bp::def("asVec3w", &::normal_cast< ::cv::Vec<unsigned char, 3>, ::cv::Vec<unsigned short, 3> >, (bp::arg("inst_Vec3b")));
        bp::def("asVec3i", &::normal_cast< ::cv::Vec<unsigned char, 3>, ::cv::Vec<int, 3> >, (bp::arg("inst_Vec3b")));
        bp::def("asVec3f", &::normal_cast< ::cv::Vec<unsigned char, 3>, ::cv::Vec<float, 3> >, (bp::arg("inst_Vec3b")));
        bp::def("asVec3d", &::normal_cast< ::cv::Vec<unsigned char, 3>, ::cv::Vec<double, 3> >, (bp::arg("inst_Vec3b")));
        bp::def("asVec3b", &::normal_cast< ::cv::Vec<short, 3>, ::cv::Vec<unsigned char, 3> >, (bp::arg("inst_Vec3s")));
        bp::def("asVec3w", &::normal_cast< ::cv::Vec<short, 3>, ::cv::Vec<unsigned short, 3> >, (bp::arg("inst_Vec3s")));
        bp::def("asVec3i", &::normal_cast< ::cv::Vec<short, 3>, ::cv::Vec<int, 3> >, (bp::arg("inst_Vec3s")));
        bp::def("asVec3f", &::normal_cast< ::cv::Vec<short, 3>, ::cv::Vec<float, 3> >, (bp::arg("inst_Vec3s")));
        bp::def("asVec3d", &::normal_cast< ::cv::Vec<short, 3>, ::cv::Vec<double, 3> >, (bp::arg("inst_Vec3s")));
        bp::def("asVec3b", &::normal_cast< ::cv::Vec<unsigned short, 3>, ::cv::Vec<unsigned char, 3> >, (bp::arg("inst_Vec3w")));
        bp::def("asVec3s", &::normal_cast< ::cv::Vec<unsigned short, 3>, ::cv::Vec<short, 3> >, (bp::arg("inst_Vec3w")));
        bp::def("asVec3i", &::normal_cast< ::cv::Vec<unsigned short, 3>, ::cv::Vec<int, 3> >, (bp::arg("inst_Vec3w")));
        bp::def("asVec3f", &::normal_cast< ::cv::Vec<unsigned short, 3>, ::cv::Vec<float, 3> >, (bp::arg("inst_Vec3w")));
        bp::def("asVec3d", &::normal_cast< ::cv::Vec<unsigned short, 3>, ::cv::Vec<double, 3> >, (bp::arg("inst_Vec3w")));
        bp::def("asVec3b", &::normal_cast< ::cv::Vec<int, 3>, ::cv::Vec<unsigned char, 3> >, (bp::arg("inst_Vec3i")));
        bp::def("asVec3s", &::normal_cast< ::cv::Vec<int, 3>, ::cv::Vec<short, 3> >, (bp::arg("inst_Vec3i")));
        bp::def("asVec3w", &::normal_cast< ::cv::Vec<int, 3>, ::cv::Vec<unsigned short, 3> >, (bp::arg("inst_Vec3i")));
        bp::def("asVec3f", &::normal_cast< ::cv::Vec<int, 3>, ::cv::Vec<float, 3> >, (bp::arg("inst_Vec3i")));
        bp::def("asVec3d", &::normal_cast< ::cv::Vec<int, 3>, ::cv::Vec<double, 3> >, (bp::arg("inst_Vec3i")));
        bp::def("asVec3b", &::normal_cast< ::cv::Vec<float, 3>, ::cv::Vec<unsigned char, 3> >, (bp::arg("inst_Vec3f")));
        bp::def("asVec3s", &::normal_cast< ::cv::Vec<float, 3>, ::cv::Vec<short, 3> >, (bp::arg("inst_Vec3f")));
        bp::def("asVec3w", &::normal_cast< ::cv::Vec<float, 3>, ::cv::Vec<unsigned short, 3> >, (bp::arg("inst_Vec3f")));
        bp::def("asVec3i", &::normal_cast< ::cv::Vec<float, 3>, ::cv::Vec<int, 3> >, (bp::arg("inst_Vec3f")));
        bp::def("asVec3d", &::normal_cast< ::cv::Vec<float, 3>, ::cv::Vec<double, 3> >, (bp::arg("inst_Vec3f")));
        bp::def("asVec3b", &::normal_cast< ::cv::Vec<double, 3>, ::cv::Vec<unsigned char, 3> >, (bp::arg("inst_Vec3d")));
        bp::def("asVec3s", &::normal_cast< ::cv::Vec<double, 3>, ::cv::Vec<short, 3> >, (bp::arg("inst_Vec3d")));
        bp::def("asVec3w", &::normal_cast< ::cv::Vec<double, 3>, ::cv::Vec<unsigned short, 3> >, (bp::arg("inst_Vec3d")));
        bp::def("asVec3i", &::normal_cast< ::cv::Vec<double, 3>, ::cv::Vec<int, 3> >, (bp::arg("inst_Vec3d")));
        bp::def("asVec3f", &::normal_cast< ::cv::Vec<double, 3>, ::cv::Vec<float, 3> >, (bp::arg("inst_Vec3d")));
        bp::def("asVec4s", &::normal_cast< ::cv::Vec<unsigned char, 4>, ::cv::Vec<short, 4> >, (bp::arg("inst_Vec4b")));
        bp::def("asVec4w", &::normal_cast< ::cv::Vec<unsigned char, 4>, ::cv::Vec<unsigned short, 4> >, (bp::arg("inst_Vec4b")));
        bp::def("asVec4i", &::normal_cast< ::cv::Vec<unsigned char, 4>, ::cv::Vec<int, 4> >, (bp::arg("inst_Vec4b")));
        bp::def("asVec4f", &::normal_cast< ::cv::Vec<unsigned char, 4>, ::cv::Vec<float, 4> >, (bp::arg("inst_Vec4b")));
        bp::def("asVec4d", &::normal_cast< ::cv::Vec<unsigned char, 4>, ::cv::Vec<double, 4> >, (bp::arg("inst_Vec4b")));
        bp::def("asVec4b", &::normal_cast< ::cv::Vec<short, 4>, ::cv::Vec<unsigned char, 4> >, (bp::arg("inst_Vec4s")));
        bp::def("asVec4w", &::normal_cast< ::cv::Vec<short, 4>, ::cv::Vec<unsigned short, 4> >, (bp::arg("inst_Vec4s")));
        bp::def("asVec4i", &::normal_cast< ::cv::Vec<short, 4>, ::cv::Vec<int, 4> >, (bp::arg("inst_Vec4s")));
        bp::def("asVec4f", &::normal_cast< ::cv::Vec<short, 4>, ::cv::Vec<float, 4> >, (bp::arg("inst_Vec4s")));
        bp::def("asVec4d", &::normal_cast< ::cv::Vec<short, 4>, ::cv::Vec<double, 4> >, (bp::arg("inst_Vec4s")));
        bp::def("asVec4b", &::normal_cast< ::cv::Vec<unsigned short, 4>, ::cv::Vec<unsigned char, 4> >, (bp::arg("inst_Vec4w")));
        bp::def("asVec4s", &::normal_cast< ::cv::Vec<unsigned short, 4>, ::cv::Vec<short, 4> >, (bp::arg("inst_Vec4w")));
        bp::def("asVec4i", &::normal_cast< ::cv::Vec<unsigned short, 4>, ::cv::Vec<int, 4> >, (bp::arg("inst_Vec4w")));
        bp::def("asVec4f", &::normal_cast< ::cv::Vec<unsigned short, 4>, ::cv::Vec<float, 4> >, (bp::arg("inst_Vec4w")));
        bp::def("asVec4d", &::normal_cast< ::cv::Vec<unsigned short, 4>, ::cv::Vec<double, 4> >, (bp::arg("inst_Vec4w")));
        bp::def("asVec4b", &::normal_cast< ::cv::Vec<int, 4>, ::cv::Vec<unsigned char, 4> >, (bp::arg("inst_Vec4i")));
        bp::def("asVec4s", &::normal_cast< ::cv::Vec<int, 4>, ::cv::Vec<short, 4> >, (bp::arg("inst_Vec4i")));
        bp::def("asVec4w", &::normal_cast< ::cv::Vec<int, 4>, ::cv::Vec<unsigned short, 4> >, (bp::arg("inst_Vec4i")));
        bp::def("asVec4f", &::normal_cast< ::cv::Vec<int, 4>, ::cv::Vec<float, 4> >, (bp::arg("inst_Vec4i")));
        bp::def("asVec4d", &::normal_cast< ::cv::Vec<int, 4>, ::cv::Vec<double, 4> >, (bp::arg("inst_Vec4i")));
        bp::def("asVec4b", &::normal_cast< ::cv::Vec<float, 4>, ::cv::Vec<unsigned char, 4> >, (bp::arg("inst_Vec4f")));
        bp::def("asVec4s", &::normal_cast< ::cv::Vec<float, 4>, ::cv::Vec<short, 4> >, (bp::arg("inst_Vec4f")));
        bp::def("asVec4w", &::normal_cast< ::cv::Vec<float, 4>, ::cv::Vec<unsigned short, 4> >, (bp::arg("inst_Vec4f")));
        bp::def("asVec4i", &::normal_cast< ::cv::Vec<float, 4>, ::cv::Vec<int, 4> >, (bp::arg("inst_Vec4f")));
        bp::def("asVec4d", &::normal_cast< ::cv::Vec<float, 4>, ::cv::Vec<double, 4> >, (bp::arg("inst_Vec4f")));
        bp::def("asVec4b", &::normal_cast< ::cv::Vec<double, 4>, ::cv::Vec<unsigned char, 4> >, (bp::arg("inst_Vec4d")));
        bp::def("asVec4s", &::normal_cast< ::cv::Vec<double, 4>, ::cv::Vec<short, 4> >, (bp::arg("inst_Vec4d")));
        bp::def("asVec4w", &::normal_cast< ::cv::Vec<double, 4>, ::cv::Vec<unsigned short, 4> >, (bp::arg("inst_Vec4d")));
        bp::def("asVec4i", &::normal_cast< ::cv::Vec<double, 4>, ::cv::Vec<int, 4> >, (bp::arg("inst_Vec4d")));
        bp::def("asVec4f", &::normal_cast< ::cv::Vec<double, 4>, ::cv::Vec<float, 4> >, (bp::arg("inst_Vec4d")));
        bp::def("asVec6d", &::normal_cast< ::cv::Vec<float, 6>, ::cv::Vec<double, 6> >, (bp::arg("inst_Vec6f")));
        bp::def("asVec6f", &::normal_cast< ::cv::Vec<double, 6>, ::cv::Vec<float, 6> >, (bp::arg("inst_Vec6d")));
        bp::def("asComplexd", &::normal_cast< ::cv::Complex<float>, ::cv::Complex<double> >, (bp::arg("inst_Complexf")));
        bp::def("asComplexf", &::normal_cast< ::cv::Complex<double>, ::cv::Complex<float> >, (bp::arg("inst_Complexd")));
        bp::def("asCvPoint", &::normal_cast< ::cv::Point_< int >, ::CvPoint >, (bp::arg("inst_Point2i")));
        bp::def("asCvPoint2D32f", &::normal_cast< ::cv::Point_< int >, ::CvPoint2D32f >, (bp::arg("inst_Point2i")));
        bp::def("asVec2i", &::normal_cast< ::cv::Point_< int >, ::cv::Vec< int, 2 > >, (bp::arg("inst_Point2i")));
        bp::def("asCvPoint", &::normal_cast< ::cv::Point_< float >, ::CvPoint >, (bp::arg("inst_Point2f")));
        bp::def("asCvPoint2D32f", &::normal_cast< ::cv::Point_< float >, ::CvPoint2D32f >, (bp::arg("inst_Point2f")));
        bp::def("asVec2f", &::normal_cast< ::cv::Point_< float >, ::cv::Vec< float, 2 > >, (bp::arg("inst_Point2f")));
        bp::def("asCvPoint", &::normal_cast< ::cv::Point_< double >, ::CvPoint >, (bp::arg("inst_Point2d")));
        bp::def("asCvPoint2D32f", &::normal_cast< ::cv::Point_< double >, ::CvPoint2D32f >, (bp::arg("inst_Point2d")));
        bp::def("asVec2d", &::normal_cast< ::cv::Point_< double >, ::cv::Vec< double, 2 > >, (bp::arg("inst_Point2d")));
        bp::def("asPoint2f", &::normal_cast< ::cv::Point_<int>, ::cv::Point_<float> >, (bp::arg("inst_Point2i")));
        bp::def("asPoint2d", &::normal_cast< ::cv::Point_<int>, ::cv::Point_<double> >, (bp::arg("inst_Point2i")));
        bp::def("asPoint2i", &::normal_cast< ::cv::Point_<float>, ::cv::Point_<int> >, (bp::arg("inst_Point2f")));
        bp::def("asPoint2d", &::normal_cast< ::cv::Point_<float>, ::cv::Point_<double> >, (bp::arg("inst_Point2f")));
        bp::def("asPoint2i", &::normal_cast< ::cv::Point_<double>, ::cv::Point_<int> >, (bp::arg("inst_Point2d")));
        bp::def("asPoint2f", &::normal_cast< ::cv::Point_<double>, ::cv::Point_<float> >, (bp::arg("inst_Point2d")));
        bp::def("asCvPoint3D32f", &::normal_cast< ::cv::Point3_< int >, ::CvPoint3D32f >, (bp::arg("inst_Point3i")));
        bp::def("asVec3i", &::normal_cast< ::cv::Point3_< int >, ::cv::Vec< int, 3 > >, (bp::arg("inst_Point3i")));
        bp::def("asCvPoint3D32f", &::normal_cast< ::cv::Point3_< float >, ::CvPoint3D32f >, (bp::arg("inst_Point3f")));
        bp::def("asVec3f", &::normal_cast< ::cv::Point3_< float >, ::cv::Vec< float, 3 > >, (bp::arg("inst_Point3f")));
        bp::def("asCvPoint3D32f", &::normal_cast< ::cv::Point3_< double >, ::CvPoint3D32f >, (bp::arg("inst_Point3d")));
        bp::def("asVec3d", &::normal_cast< ::cv::Point3_< double >, ::cv::Vec< double, 3 > >, (bp::arg("inst_Point3d")));
        bp::def("asPoint3f", &::normal_cast< ::cv::Point3_<int>, ::cv::Point3_<float> >, (bp::arg("inst_Point3i")));
        bp::def("asPoint3d", &::normal_cast< ::cv::Point3_<int>, ::cv::Point3_<double> >, (bp::arg("inst_Point3i")));
        bp::def("asPoint3i", &::normal_cast< ::cv::Point3_<float>, ::cv::Point3_<int> >, (bp::arg("inst_Point3f")));
        bp::def("asPoint3d", &::normal_cast< ::cv::Point3_<float>, ::cv::Point3_<double> >, (bp::arg("inst_Point3f")));
        bp::def("asPoint3i", &::normal_cast< ::cv::Point3_<double>, ::cv::Point3_<int> >, (bp::arg("inst_Point3d")));
        bp::def("asPoint3f", &::normal_cast< ::cv::Point3_<double>, ::cv::Point3_<float> >, (bp::arg("inst_Point3d")));
        bp::def("asCvSize", &::normal_cast< ::cv::Size_< int >, ::CvSize >, (bp::arg("inst_Size2i")));
        bp::def("asCvSize2D32f", &::normal_cast< ::cv::Size_< int >, ::CvSize2D32f >, (bp::arg("inst_Size2i")));
        bp::def("asCvSize", &::normal_cast< ::cv::Size_< float >, ::CvSize >, (bp::arg("inst_Size2f")));
        bp::def("asCvSize2D32f", &::normal_cast< ::cv::Size_< float >, ::CvSize2D32f >, (bp::arg("inst_Size2f")));
        bp::def("asSize2f", &::normal_cast< ::cv::Size_<int>, ::cv::Size_<float> >, (bp::arg("inst_Size2i")));
        bp::def("asSize2i", &::normal_cast< ::cv::Size_<float>, ::cv::Size_<int> >, (bp::arg("inst_Size2f")));
        bp::def("asCvRect", &::normal_cast< ::cv::Rect_< int >, ::CvRect >, (bp::arg("inst_Rect")));
        bp::def("asCvBox2D", &::normal_cast< ::cv::RotatedRect, ::CvBox2D >, (bp::arg("inst_RotatedRect")));
        bp::def("asCvScalar", &::normal_cast< ::cv::Scalar_< double >, ::CvScalar >, (bp::arg("inst_Scalar")));
        bp::def("asCvSlice", &::normal_cast< ::cv::Range, ::CvSlice >, (bp::arg("inst_Range")));
        bp::def("asCvTermCriteria", &::normal_cast< ::cv::TermCriteria, ::CvTermCriteria >, (bp::arg("inst_TermCriteria")));;
    }

}
