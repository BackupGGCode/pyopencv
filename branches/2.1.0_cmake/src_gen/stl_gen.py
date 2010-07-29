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

# register std::vector<std::string>
# sb.expose_class_vector('std::string')
# sb.register_decl('str', 'std::string')

# register vectors of basic data types
sb.expose_class_vector('char')
sb.expose_class_vector('unsigned char')
sb.expose_class_vector('short')
sb.expose_class_vector('unsigned short')
sb.expose_class_vector('int')
sb.expose_class_vector('unsigned int')
sb.expose_class_vector('long')
sb.expose_class_vector('unsigned long')
sb.expose_class_vector('long long')
sb.expose_class_vector('unsigned long long')
sb.expose_class_vector('float')
sb.expose_class_vector('double')
sb.expose_class_vector('unsigned char *', 'vector_string')
sb.expose_class_vector('std::vector< int >')
sb.expose_class_vector('std::vector< float >')

z = sb.dummy_struct
z.include_files.append("opencv_converters.hpp")
z.include_files.append("sequence.hpp")
sb.add_reg_code("sdcpp::register_sdobject<sdcpp::sequence>();")

sb.done()
