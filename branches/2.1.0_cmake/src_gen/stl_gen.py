#!/usr/bin/env python
# PyOpencv - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

import sdpypp
sb = sdpypp.SdModuleBuilder('stl')

# basic data types
sb.register_decl('None', 'void')
sb.register_decl('bool', 'bool')
sb.register_decl('int8', 'char')
sb.register_decl('int8', 'signed char')
sb.register_decl('int8', 'schar')
sb.register_decl('uint8', 'unsigned char')
sb.register_decl('uint8', 'uchar')
sb.register_decl('int16', 'short')
sb.register_decl('int16', 'short int')
sb.register_decl('uint16', 'unsigned short')
sb.register_decl('uint16', 'short unsigned int')
sb.register_decl('uint16', 'ushort')
sb.register_decl('int', 'int')
sb.register_decl('uint', 'unsigned int')
sb.register_decl('long', 'long')
sb.register_decl('ulong', 'unsigned long')
sb.register_decl('float32', 'float')
sb.register_decl('float64', 'double')

# register std::vector<std::string>
# sb.register_vec('std::vector', 'std::string')
# sb.register_decl('str', 'std::string')

# register vectors of basic data types
sb.register_vec('std::vector', 'char', 'vector_int8', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned char', 'vector_uint8', pyEquivName='Mat')
sb.register_vec('std::vector', 'short', 'vector_int16', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned short', 'vector_uint16', pyEquivName='Mat')
sb.register_vec('std::vector', 'int', 'vector_int', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned int', 'vector_uint', pyEquivName='Mat')
sb.register_vec('std::vector', 'long', 'vector_long', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned long', 'vector_ulong', pyEquivName='Mat')
sb.register_vec('std::vector', 'long long', 'vector_int64', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned long long', 'vector_uint64', pyEquivName='Mat')
sb.register_vec('std::vector', 'float', 'vector_float32', pyEquivName='Mat')
sb.register_vec('std::vector', 'double', 'vector_float64', pyEquivName='Mat')
sb.register_vec('std::vector', 'unsigned char *', 'vector_string')
sb.register_vec('std::vector', 'std::vector< int >', 'vector_vector_int')
sb.register_vec('std::vector', 'std::vector< float >', 'vector_vector_float32')

z = sb.dummy_struct
z.include_files.append("opencv_converters.hpp")
z.include_files.append("sequence.hpp")
sb.add_reg_code("sdcpp::register_sdobject<sdcpp::sequence>();")

sb.done()
