#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.0 using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Subroutines related to writing to the opencv2_include\template_instantiations.h file
# -----------------------------------------------------------------------------------------------
ti = []

def add_ti(class_, element_type, new_name=None):
    global ti
    if new_name is None:
        new_name = class_ + '_' + element_type
    ti.append((class_, element_type, new_name))



def generate_ti():
    # template instantiations
    basic_types = ('bool', 'uchar', 'schar', 'ushort', 'short', 'int', 'float', 'double')
    for z1 in ('DataDepth', 'DataType'):
        for z2 in basic_types:
            add_ti(z1, z2)
    add_ti('DataType', 'Range')

    add_ti('Rect_', 'int', 'Rect')
    add_ti('Rect_', 'float', 'Rectf')
    add_ti('Rect_', 'double', 'Rectd')

    add_ti('Size_', 'int', 'Size')
    add_ti('Size_', 'int', 'Size2i')
    add_ti('Size_', 'float', 'Size2f')
    add_ti('Size_', 'double', 'Size2d')

    add_ti('Complex', 'float', 'Complexf')
    add_ti('Complex', 'double', 'Complexd')

    add_ti('Point_', 'int', 'Point')
    add_ti('Point_', 'int', 'Point2i')
    add_ti('Point_', 'float', 'Point2f')
    add_ti('Point_', 'double', 'Point2d')

    add_ti('Point3_', 'int', 'Point3i')
    add_ti('Point3_', 'float', 'Point3f')
    add_ti('Point3_', 'double', 'Point3d')

    add_ti('Scalar_', 'double', 'Scalar')

    add_ti('Vec', 'uchar, 2', 'Vec2b')
    add_ti('Vec', 'uchar, 3', 'Vec3b')
    add_ti('Vec', 'uchar, 4', 'Vec4b')

    add_ti('Vec', 'short, 2', 'Vec2s')
    add_ti('Vec', 'short, 3', 'Vec3s')
    add_ti('Vec', 'short, 4', 'Vec4s')

    add_ti('Vec', 'ushort, 2', 'Vec2w')
    add_ti('Vec', 'ushort, 3', 'Vec3w')
    add_ti('Vec', 'ushort, 4', 'Vec4w')

    add_ti('Vec', 'int, 2', 'Vec2i')
    add_ti('Vec', 'int, 3', 'Vec3i')
    add_ti('Vec', 'int, 4', 'Vec4i')

    add_ti('Vec', 'float, 2', 'Vec2f')
    add_ti('Vec', 'float, 3', 'Vec3f')
    add_ti('Vec', 'float, 4', 'Vec4f')
    add_ti('Vec', 'float, 6', 'Vec6f')

    add_ti('Vec', 'double, 2', 'Vec2d')
    add_ti('Vec', 'double, 3', 'Vec3d')
    add_ti('Vec', 'double, 4', 'Vec4d')
    add_ti('Vec', 'double, 6', 'Vec6d')
    
    add_ti('Ptr', 'FilterEngine')
    
    add_ti('vector', 'char', 'vector_int8')
    add_ti('vector', 'unsigned char', 'vector_uint8')
    add_ti('vector', 'short', 'vector_int16')
    add_ti('vector', 'unsigned short', 'vector_uint16')
    add_ti('vector', 'int', 'vector_int')
    add_ti('vector', 'unsigned int', 'vector_uint')
    add_ti('vector', 'long', 'vector_int32')
    add_ti('vector', 'unsigned long', 'vector_uint32')
    add_ti('vector', 'long long', 'vector_int64')
    add_ti('vector', 'unsigned long long', 'vector_uint64')
    add_ti('vector', 'float', 'vector_float32')
    add_ti('vector', 'double', 'vector_float64')
    
    add_ti('vector', 'Vec2i', 'vector_Vec2i')
    add_ti('vector', 'Vec2f', 'vector_Vec2f')
    add_ti('vector', 'Vec3f', 'vector_Vec3f')
    add_ti('vector', 'Vec4i', 'vector_Vec4i')
    add_ti('vector', 'Point2i', 'vector_Point')
    add_ti('vector', 'Point2f', 'vector_Point2f')
    add_ti('vector', 'Point3i', 'vector_Point3')
    add_ti('vector', 'Point3f', 'vector_Point3f')
    add_ti('vector', 'Mat', 'vector_Mat')
    add_ti('vector', 'MatND', 'vector_MatND')
    add_ti('vector', 'KeyPoint', 'vector_KeyPoint')
    add_ti('vector', 'CascadeClassifier::DTreeNode', 'vector_CascadeClassifier_DTreeNode')
    add_ti('vector', 'CascadeClassifier::DTree', 'vector_CascadeClassifier_DTree')
    add_ti('vector', 'CascadeClassifier::Stage', 'vector_CascadeClassifier_Stage')
    add_ti('vector', 'FernClassifier::Feature', 'vector_FernClassifier_Feature')
    add_ti('vector', 'Octree::Node', 'vector_Octree_Node')
    add_ti('vector', 'CvFuzzyRule*', 'vector_CvFuzzyRule_Ptr')
    add_ti('vector', 'CvFuzzyCurve', 'vector_CvFuzzyCurve')
    add_ti('vector', 'CvFuzzyPoint', 'vector_CvFuzzyPoint')
    add_ti('vector', 'unsigned char *', 'vector_string')
    add_ti('vector', 'KDTree::Node', 'vector_KDTree_Node')
    add_ti('vector', 'vector_int', 'vector_vector_int')
    add_ti('vector', 'vector_float32', 'vector_vector_float32')
    add_ti('vector', 'vector_Point', 'vector_vector_Point')
    add_ti('vector', 'vector_Point2f', 'vector_vector_Point2f')
    add_ti('vector', 'vector_Point3f', 'vector_vector_Point3f')
    add_ti('vector', 'vector_Vec2i', 'vector_vector_Vec2i')
    add_ti('vector', 'Rect', 'vector_Rect')


def finalize_ti():
    tif = open('../src/pyopencvext/template_instantiations.hpp', 'wt')
    tif.write('''#ifndef SD_TEMPLATE_INSTANTIATIONS_H
#define SD_TEMPlATE_INSTANTIATIONS_H

namespace cv {
''')

    for z in ti:
        tif.write('''
    #ifndef %s
    typedef %s < %s > %s;
    #endif

''' % (z[2], z[0], z[1], z[2]))

    tif.write('''

    struct __dummy_struct {
''')

    for i in xrange(len(ti)):
        tif.write('         %s var%d;\n' % (ti[i][2], i))

    tif.write('''    };
}

#endif
''')


# Finalize template instantiations
generate_ti()
finalize_ti()



