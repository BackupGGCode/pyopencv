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

def generate_code(mb, cc, D, FT, CP):
    cc.write('''
#=============================================================================
# sdopencv
#=============================================================================


    ''')


    sdopencv = mb.namespace('sdopencv')
    sdopencv.include()
    
    for t in ('DifferentialImage', 'IntegralImage'):
        z = sdopencv.class_(t)
        mb.init_class(z)
        mb.finalize_class(z)

    # IntegralHistogram
    z = sdopencv.class_('IntegralHistogram')
    z.include_files.append("opencv_converters.hpp")
    mb.init_class(z)
    z.constructor(lambda x: len(x.arguments) > 1).exclude()
    z.add_declaration_code('''
static boost::shared_ptr<sdopencv::IntegralHistogram> IntegralHistogram__init1__(int histSize, bp::sequence const &ranges, bool uniform)
{
    std::vector<float> ranges2; convert_seq_to_vector(ranges, ranges2);
    return boost::shared_ptr<sdopencv::IntegralHistogram>(new sdopencv::IntegralHistogram(histSize, ranges2, uniform));
}

    ''')
    z.add_registration_code('def("__init__", bp::make_constructor(&IntegralHistogram__init1__, bp::default_call_policies(), ( bp::arg("histSize"), bp::arg("ranges"), bp::arg("uniform")=bp::object(true) )))')

    mb.finalize_class(z)
