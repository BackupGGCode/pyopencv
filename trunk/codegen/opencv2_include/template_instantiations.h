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


    #ifndef vector_bool
    typedef vector < bool > vector_bool;
    #endif


    #ifndef vector_uchar
    typedef vector < uchar > vector_uchar;
    #endif


    #ifndef vector_schar
    typedef vector < schar > vector_schar;
    #endif


    #ifndef vector_ushort
    typedef vector < ushort > vector_ushort;
    #endif


    #ifndef vector_short
    typedef vector < short > vector_short;
    #endif


    #ifndef vector_int
    typedef vector < int > vector_int;
    #endif


    #ifndef vector_float
    typedef vector < float > vector_float;
    #endif


    #ifndef vector_double
    typedef vector < double > vector_double;
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
         vector_bool var16;
         vector_uchar var17;
         vector_schar var18;
         vector_ushort var19;
         vector_short var20;
         vector_int var21;
         vector_float var22;
         vector_double var23;
         DataType_Range var24;
         Rect var25;
         Rectf var26;
         Rectd var27;
         Size var28;
         Size2i var29;
         Size2f var30;
         Size2d var31;
         Complexf var32;
         Complexd var33;
         Point var34;
         Point2i var35;
         Point2f var36;
         Point2d var37;
         Point3i var38;
         Point3f var39;
         Point3d var40;
         Scalar var41;
         Vec2b var42;
         Vec3b var43;
         Vec4b var44;
         Vec2s var45;
         Vec3s var46;
         Vec4s var47;
         Vec2w var48;
         Vec3w var49;
         Vec4w var50;
         Vec2i var51;
         Vec3i var52;
         Vec4i var53;
         Vec2f var54;
         Vec3f var55;
         Vec4f var56;
         Vec6f var57;
         Vec2d var58;
         Vec3d var59;
         Vec4d var60;
         Vec6d var61;
    };
}

#endif
