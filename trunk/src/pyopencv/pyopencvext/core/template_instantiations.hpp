#ifndef SD_TEMPLATE_INSTANTIATIONS_H
#define SD_TEMPlATE_INSTANTIATIONS_H

namespace cv {

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
        vector_int8 var0;
        vector_uint8 var1;
        vector_int16 var2;
        vector_uint16 var3;
        vector_int var4;
        vector_uint var5;
        vector_long var6;
        vector_ulong var7;
        vector_int64 var8;
        vector_uint64 var9;
        vector_float32 var10;
        vector_float64 var11;
        vector_Vec2i var12;
        vector_Vec2f var13;
        vector_Vec3f var14;
        vector_Vec4i var15;
        vector_Point var16;
        vector_Point2f var17;
        vector_Point3 var18;
        vector_Point3f var19;
        vector_Mat var20;
        vector_MatND var21;
        vector_KeyPoint var22;
        vector_CascadeClassifier_DTreeNode var23;
        vector_CascadeClassifier_DTree var24;
        vector_CascadeClassifier_Stage var25;
        vector_FernClassifier_Feature var26;
        vector_Octree_Node var27;
        vector_CvFuzzyRule_Ptr var28;
        vector_CvFuzzyCurve var29;
        vector_CvFuzzyPoint var30;
        vector_string var31;
        vector_KDTree_Node var32;
        vector_vector_int var33;
        vector_vector_float32 var34;
        vector_vector_Point var35;
        vector_vector_Point2f var36;
        vector_vector_Point3f var37;
        vector_vector_Vec2i var38;
        vector_Rect var39;
    };
}

#endif
