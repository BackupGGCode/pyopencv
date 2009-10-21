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



    CV_INLINE void __dummy_func(
        int __dummy_var = 0
            + sizeof(DataDepth_bool)
            + sizeof(DataDepth_uchar)
            + sizeof(DataDepth_schar)
            + sizeof(DataDepth_ushort)
            + sizeof(DataDepth_short)
            + sizeof(DataDepth_int)
            + sizeof(DataDepth_float)
            + sizeof(DataDepth_double)
            + sizeof(DataType_bool)
            + sizeof(DataType_uchar)
            + sizeof(DataType_schar)
            + sizeof(DataType_ushort)
            + sizeof(DataType_short)
            + sizeof(DataType_int)
            + sizeof(DataType_float)
            + sizeof(DataType_double)
            + sizeof(DataType_Range)
            + sizeof(Rect)
            + sizeof(Rectf)
            + sizeof(Rectd)
            + sizeof(Size)
            + sizeof(Size2i)
            + sizeof(Size2f)
            + sizeof(Size2d)
            + sizeof(Complexf)
            + sizeof(Complexd)
            + sizeof(Point)
            + sizeof(Point2i)
            + sizeof(Point2f)
            + sizeof(Point2d)
            + sizeof(Point3i)
            + sizeof(Point3f)
            + sizeof(Point3d)
            + sizeof(Scalar)
            + sizeof(Vec2b)
            + sizeof(Vec3b)
            + sizeof(Vec4b)
            + sizeof(Vec2s)
            + sizeof(Vec3s)
            + sizeof(Vec4s)
            + sizeof(Vec2w)
            + sizeof(Vec3w)
            + sizeof(Vec4w)
            + sizeof(Vec2i)
            + sizeof(Vec3i)
            + sizeof(Vec4i)
            + sizeof(Vec2f)
            + sizeof(Vec3f)
            + sizeof(Vec4f)
            + sizeof(Vec6f)
            + sizeof(Vec2d)
            + sizeof(Vec3d)
            + sizeof(Vec4d)
            + sizeof(Vec6d)
    ) {}
}

#endif
