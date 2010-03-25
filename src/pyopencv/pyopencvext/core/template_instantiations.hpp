#ifndef SD_TEMPLATE_INSTANTIATIONS_H
#define SD_TEMPlATE_INSTANTIATIONS_H

namespace cv {

    #ifndef DataDepth_bool
    typedef DataDepth < bool > DataDepth_bool;
    #endif


    #ifndef DataDepth_uchar
    typedef DataDepth < uchar > DataDepth_uchar;
    #endif


    #ifndef DataDepth_schar
    typedef DataDepth < schar > DataDepth_schar;
    #endif


    #ifndef DataDepth_ushort
    typedef DataDepth < ushort > DataDepth_ushort;
    #endif


    #ifndef DataDepth_short
    typedef DataDepth < short > DataDepth_short;
    #endif


    #ifndef DataDepth_int
    typedef DataDepth < int > DataDepth_int;
    #endif


    #ifndef DataDepth_float
    typedef DataDepth < float > DataDepth_float;
    #endif


    #ifndef DataDepth_double
    typedef DataDepth < double > DataDepth_double;
    #endif


    #ifndef DataType_bool
    typedef DataType < bool > DataType_bool;
    #endif


    #ifndef DataType_uchar
    typedef DataType < uchar > DataType_uchar;
    #endif


    #ifndef DataType_schar
    typedef DataType < schar > DataType_schar;
    #endif


    #ifndef DataType_ushort
    typedef DataType < ushort > DataType_ushort;
    #endif


    #ifndef DataType_short
    typedef DataType < short > DataType_short;
    #endif


    #ifndef DataType_int
    typedef DataType < int > DataType_int;
    #endif


    #ifndef DataType_float
    typedef DataType < float > DataType_float;
    #endif


    #ifndef DataType_double
    typedef DataType < double > DataType_double;
    #endif


    #ifndef DataType_Range
    typedef DataType < Range > DataType_Range;
    #endif


    #ifndef Rect
    typedef Rect_ < int > Rect;
    #endif


    #ifndef Rectf
    typedef Rect_ < float > Rectf;
    #endif


    #ifndef Rectd
    typedef Rect_ < double > Rectd;
    #endif


    #ifndef Size
    typedef Size_ < int > Size;
    #endif


    #ifndef Size2i
    typedef Size_ < int > Size2i;
    #endif


    #ifndef Size2f
    typedef Size_ < float > Size2f;
    #endif


    #ifndef Size2d
    typedef Size_ < double > Size2d;
    #endif


    #ifndef Complexf
    typedef Complex < float > Complexf;
    #endif


    #ifndef Complexd
    typedef Complex < double > Complexd;
    #endif


    #ifndef Point
    typedef Point_ < int > Point;
    #endif


    #ifndef Point2i
    typedef Point_ < int > Point2i;
    #endif


    #ifndef Point2f
    typedef Point_ < float > Point2f;
    #endif


    #ifndef Point2d
    typedef Point_ < double > Point2d;
    #endif


    #ifndef Point3i
    typedef Point3_ < int > Point3i;
    #endif


    #ifndef Point3f
    typedef Point3_ < float > Point3f;
    #endif


    #ifndef Point3d
    typedef Point3_ < double > Point3d;
    #endif


    #ifndef Scalar
    typedef Scalar_ < double > Scalar;
    #endif


    #ifndef Vec2b
    typedef Vec < uchar, 2 > Vec2b;
    #endif


    #ifndef Vec3b
    typedef Vec < uchar, 3 > Vec3b;
    #endif


    #ifndef Vec4b
    typedef Vec < uchar, 4 > Vec4b;
    #endif


    #ifndef Vec2s
    typedef Vec < short, 2 > Vec2s;
    #endif


    #ifndef Vec3s
    typedef Vec < short, 3 > Vec3s;
    #endif


    #ifndef Vec4s
    typedef Vec < short, 4 > Vec4s;
    #endif


    #ifndef Vec2w
    typedef Vec < ushort, 2 > Vec2w;
    #endif


    #ifndef Vec3w
    typedef Vec < ushort, 3 > Vec3w;
    #endif


    #ifndef Vec4w
    typedef Vec < ushort, 4 > Vec4w;
    #endif


    #ifndef Vec2i
    typedef Vec < int, 2 > Vec2i;
    #endif


    #ifndef Vec3i
    typedef Vec < int, 3 > Vec3i;
    #endif


    #ifndef Vec4i
    typedef Vec < int, 4 > Vec4i;
    #endif


    #ifndef Vec2f
    typedef Vec < float, 2 > Vec2f;
    #endif


    #ifndef Vec3f
    typedef Vec < float, 3 > Vec3f;
    #endif


    #ifndef Vec4f
    typedef Vec < float, 4 > Vec4f;
    #endif


    #ifndef Vec6f
    typedef Vec < float, 6 > Vec6f;
    #endif


    #ifndef Vec2d
    typedef Vec < double, 2 > Vec2d;
    #endif


    #ifndef Vec3d
    typedef Vec < double, 3 > Vec3d;
    #endif


    #ifndef Vec4d
    typedef Vec < double, 4 > Vec4d;
    #endif


    #ifndef Vec6d
    typedef Vec < double, 6 > Vec6d;
    #endif


    #ifndef Ptr_FilterEngine
    typedef Ptr < FilterEngine > Ptr_FilterEngine;
    #endif


    #ifndef vector_int8
    typedef vector < char > vector_int8;
    #endif


    #ifndef vector_uint8
    typedef vector < unsigned char > vector_uint8;
    #endif


    #ifndef vector_int16
    typedef vector < short > vector_int16;
    #endif


    #ifndef vector_uint16
    typedef vector < unsigned short > vector_uint16;
    #endif


    #ifndef vector_int
    typedef vector < int > vector_int;
    #endif


    #ifndef vector_uint
    typedef vector < unsigned int > vector_uint;
    #endif


    #ifndef vector_int32
    typedef vector < long > vector_int32;
    #endif


    #ifndef vector_uint32
    typedef vector < unsigned long > vector_uint32;
    #endif


    #ifndef vector_int64
    typedef vector < long long > vector_int64;
    #endif


    #ifndef vector_uint64
    typedef vector < unsigned long long > vector_uint64;
    #endif


    #ifndef vector_float32
    typedef vector < float > vector_float32;
    #endif


    #ifndef vector_float64
    typedef vector < double > vector_float64;
    #endif


    #ifndef vector_Vec2i
    typedef vector < Vec2i > vector_Vec2i;
    #endif


    #ifndef vector_Vec2f
    typedef vector < Vec2f > vector_Vec2f;
    #endif


    #ifndef vector_Vec3f
    typedef vector < Vec3f > vector_Vec3f;
    #endif


    #ifndef vector_Vec4i
    typedef vector < Vec4i > vector_Vec4i;
    #endif


    #ifndef vector_Point
    typedef vector < Point2i > vector_Point;
    #endif


    #ifndef vector_Point2f
    typedef vector < Point2f > vector_Point2f;
    #endif


    #ifndef vector_Point3
    typedef vector < Point3i > vector_Point3;
    #endif


    #ifndef vector_Point3f
    typedef vector < Point3f > vector_Point3f;
    #endif


    #ifndef vector_Mat
    typedef vector < Mat > vector_Mat;
    #endif


    #ifndef vector_MatND
    typedef vector < MatND > vector_MatND;
    #endif


    #ifndef vector_KeyPoint
    typedef vector < KeyPoint > vector_KeyPoint;
    #endif


    #ifndef vector_CascadeClassifier_DTreeNode
    typedef vector < CascadeClassifier::DTreeNode > vector_CascadeClassifier_DTreeNode;
    #endif


    #ifndef vector_CascadeClassifier_DTree
    typedef vector < CascadeClassifier::DTree > vector_CascadeClassifier_DTree;
    #endif


    #ifndef vector_CascadeClassifier_Stage
    typedef vector < CascadeClassifier::Stage > vector_CascadeClassifier_Stage;
    #endif


    #ifndef vector_FernClassifier_Feature
    typedef vector < FernClassifier::Feature > vector_FernClassifier_Feature;
    #endif


    #ifndef Ptr_Mat
    typedef Ptr < Mat > Ptr_Mat;
    #endif


    #ifndef vector_Ptr_Mat
    typedef vector < Ptr_Mat > vector_Ptr_Mat;
    #endif


    #ifndef vector_Octree_Node
    typedef vector < Octree::Node > vector_Octree_Node;
    #endif


    #ifndef vector_CvFuzzyRule_Ptr
    typedef vector < CvFuzzyRule* > vector_CvFuzzyRule_Ptr;
    #endif


    #ifndef vector_CvFuzzyCurve
    typedef vector < CvFuzzyCurve > vector_CvFuzzyCurve;
    #endif


    #ifndef vector_CvFuzzyPoint
    typedef vector < CvFuzzyPoint > vector_CvFuzzyPoint;
    #endif


    #ifndef vector_string
    typedef vector < unsigned char * > vector_string;
    #endif


    #ifndef vector_KDTree_Node
    typedef vector < KDTree::Node > vector_KDTree_Node;
    #endif


    #ifndef vector_vector_int
    typedef vector < vector_int > vector_vector_int;
    #endif


    #ifndef vector_vector_float32
    typedef vector < vector_float32 > vector_vector_float32;
    #endif


    #ifndef vector_vector_Point
    typedef vector < vector_Point > vector_vector_Point;
    #endif


    #ifndef vector_vector_Point2f
    typedef vector < vector_Point2f > vector_vector_Point2f;
    #endif


    #ifndef vector_vector_Point3f
    typedef vector < vector_Point3f > vector_vector_Point3f;
    #endif


    #ifndef vector_vector_Vec2i
    typedef vector < vector_Vec2i > vector_vector_Vec2i;
    #endif


    #ifndef vector_Rect
    typedef vector < Rect > vector_Rect;
    #endif



    struct __dummy_struct {
         DataDepth_bool var0;
         DataDepth_uchar var1;
         DataDepth_schar var2;
         DataDepth_ushort var3;
         DataDepth_short var4;
         DataDepth_int var5;
         DataDepth_float var6;
         DataDepth_double var7;
         DataType_bool var8;
         DataType_uchar var9;
         DataType_schar var10;
         DataType_ushort var11;
         DataType_short var12;
         DataType_int var13;
         DataType_float var14;
         DataType_double var15;
         DataType_Range var16;
         Rect var17;
         Rectf var18;
         Rectd var19;
         Size var20;
         Size2i var21;
         Size2f var22;
         Size2d var23;
         Complexf var24;
         Complexd var25;
         Point var26;
         Point2i var27;
         Point2f var28;
         Point2d var29;
         Point3i var30;
         Point3f var31;
         Point3d var32;
         Scalar var33;
         Vec2b var34;
         Vec3b var35;
         Vec4b var36;
         Vec2s var37;
         Vec3s var38;
         Vec4s var39;
         Vec2w var40;
         Vec3w var41;
         Vec4w var42;
         Vec2i var43;
         Vec3i var44;
         Vec4i var45;
         Vec2f var46;
         Vec3f var47;
         Vec4f var48;
         Vec6f var49;
         Vec2d var50;
         Vec3d var51;
         Vec4d var52;
         Vec6d var53;
         Ptr_FilterEngine var54;
         vector_int8 var55;
         vector_uint8 var56;
         vector_int16 var57;
         vector_uint16 var58;
         vector_int var59;
         vector_uint var60;
         vector_int32 var61;
         vector_uint32 var62;
         vector_int64 var63;
         vector_uint64 var64;
         vector_float32 var65;
         vector_float64 var66;
         vector_Vec2i var67;
         vector_Vec2f var68;
         vector_Vec3f var69;
         vector_Vec4i var70;
         vector_Point var71;
         vector_Point2f var72;
         vector_Point3 var73;
         vector_Point3f var74;
         vector_Mat var75;
         vector_MatND var76;
         vector_KeyPoint var77;
         vector_CascadeClassifier_DTreeNode var78;
         vector_CascadeClassifier_DTree var79;
         vector_CascadeClassifier_Stage var80;
         vector_FernClassifier_Feature var81;
         Ptr_Mat var82;
         vector_Ptr_Mat var83;
         vector_Octree_Node var84;
         vector_CvFuzzyRule_Ptr var85;
         vector_CvFuzzyCurve var86;
         vector_CvFuzzyPoint var87;
         vector_string var88;
         vector_KDTree_Node var89;
         vector_vector_int var90;
         vector_vector_float32 var91;
         vector_vector_Point var92;
         vector_vector_Point2f var93;
         vector_vector_Point3f var94;
         vector_vector_Vec2i var95;
         vector_Rect var96;
    };
}

#endif
