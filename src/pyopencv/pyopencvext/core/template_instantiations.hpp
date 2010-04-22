#ifndef SD_TEMPLATE_INSTANTIATIONS_H
#define SD_TEMPlATE_INSTANTIATIONS_H

namespace cv {

    #ifndef Mat1b
    typedef Mat_ < uchar > Mat1b;
    #endif


    #ifndef Mat2b
    typedef Mat_ < Vec2b > Mat2b;
    #endif


    #ifndef Mat3b
    typedef Mat_ < Vec3b > Mat3b;
    #endif


    #ifndef Mat4b
    typedef Mat_ < Vec4b > Mat4b;
    #endif


    #ifndef Mat1s
    typedef Mat_ < short > Mat1s;
    #endif


    #ifndef Mat2s
    typedef Mat_ < Vec2s > Mat2s;
    #endif


    #ifndef Mat3s
    typedef Mat_ < Vec3s > Mat3s;
    #endif


    #ifndef Mat4s
    typedef Mat_ < Vec4s > Mat4s;
    #endif


    #ifndef Mat1w
    typedef Mat_ < ushort > Mat1w;
    #endif


    #ifndef Mat2w
    typedef Mat_ < Vec2w > Mat2w;
    #endif


    #ifndef Mat3w
    typedef Mat_ < Vec3w > Mat3w;
    #endif


    #ifndef Mat4w
    typedef Mat_ < Vec4w > Mat4w;
    #endif


    #ifndef Mat1i
    typedef Mat_ < int > Mat1i;
    #endif


    #ifndef Mat2i
    typedef Mat_ < Vec2i > Mat2i;
    #endif


    #ifndef Mat3i
    typedef Mat_ < Vec3i > Mat3i;
    #endif


    #ifndef Mat4i
    typedef Mat_ < Vec4i > Mat4i;
    #endif


    #ifndef Mat1f
    typedef Mat_ < float > Mat1f;
    #endif


    #ifndef Mat2f
    typedef Mat_ < Vec2f > Mat2f;
    #endif


    #ifndef Mat3f
    typedef Mat_ < Vec3f > Mat3f;
    #endif


    #ifndef Mat4f
    typedef Mat_ < Vec4f > Mat4f;
    #endif


    #ifndef Mat1d
    typedef Mat_ < double > Mat1d;
    #endif


    #ifndef Mat2d
    typedef Mat_ < Vec2d > Mat2d;
    #endif


    #ifndef Mat3d
    typedef Mat_ < Vec3d > Mat3d;
    #endif


    #ifndef Mat4d
    typedef Mat_ < Vec4d > Mat4d;
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


    #ifndef vector_long
    typedef vector < long > vector_long;
    #endif


    #ifndef vector_ulong
    typedef vector < unsigned long > vector_ulong;
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



    struct dummy_struct {
        struct dummy_struct2 {};
        Mat1b var0;
        Mat2b var1;
        Mat3b var2;
        Mat4b var3;
        Mat1s var4;
        Mat2s var5;
        Mat3s var6;
        Mat4s var7;
        Mat1w var8;
        Mat2w var9;
        Mat3w var10;
        Mat4w var11;
        Mat1i var12;
        Mat2i var13;
        Mat3i var14;
        Mat4i var15;
        Mat1f var16;
        Mat2f var17;
        Mat3f var18;
        Mat4f var19;
        Mat1d var20;
        Mat2d var21;
        Mat3d var22;
        Mat4d var23;
        vector_int8 var24;
        vector_uint8 var25;
        vector_int16 var26;
        vector_uint16 var27;
        vector_int var28;
        vector_uint var29;
        vector_long var30;
        vector_ulong var31;
        vector_int64 var32;
        vector_uint64 var33;
        vector_float32 var34;
        vector_float64 var35;
        vector_Vec2i var36;
        vector_Vec2f var37;
        vector_Vec3f var38;
        vector_Vec4i var39;
        vector_Point var40;
        vector_Point2f var41;
        vector_Point3 var42;
        vector_Point3f var43;
        vector_Mat var44;
        vector_MatND var45;
        vector_KeyPoint var46;
        vector_CascadeClassifier_DTreeNode var47;
        vector_CascadeClassifier_DTree var48;
        vector_CascadeClassifier_Stage var49;
        vector_FernClassifier_Feature var50;
        vector_Octree_Node var51;
        vector_CvFuzzyRule_Ptr var52;
        vector_CvFuzzyCurve var53;
        vector_CvFuzzyPoint var54;
        vector_string var55;
        vector_KDTree_Node var56;
        vector_vector_int var57;
        vector_vector_float32 var58;
        vector_vector_Point var59;
        vector_vector_Point2f var60;
        vector_vector_Point3f var61;
        vector_vector_Vec2i var62;
        vector_Rect var63;
    };
}

#endif
