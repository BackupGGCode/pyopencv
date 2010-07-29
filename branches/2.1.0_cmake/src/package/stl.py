#!/usr/bin/env python
# PyOpenCV - A Python wrapper for OpenCV 2.x using Boost.Python and NumPy

# Copyright (c) 2009, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of pyopencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

import common as _c
import stl_ext as _ext
from stl_ext import *
        
vector_int8.__old_init__ = vector_int8.__init__
vector_int8.__init__ = _c.__vector__init__
vector_int8.create = _c.__vector_create
vector_int8.__repr__ = _c.__vector__repr__
vector_int8.tolist = _c.__vector_tolist
vector_int8.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_int8()
_z.resize(1)
vector_int8.elem_type = _z[0].__class__
del(_z)
        
vector_uint8.__old_init__ = vector_uint8.__init__
vector_uint8.__init__ = _c.__vector__init__
vector_uint8.create = _c.__vector_create
vector_uint8.__repr__ = _c.__vector__repr__
vector_uint8.tolist = _c.__vector_tolist
vector_uint8.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_uint8()
_z.resize(1)
vector_uint8.elem_type = _z[0].__class__
del(_z)
        
vector_int16.__old_init__ = vector_int16.__init__
vector_int16.__init__ = _c.__vector__init__
vector_int16.create = _c.__vector_create
vector_int16.__repr__ = _c.__vector__repr__
vector_int16.tolist = _c.__vector_tolist
vector_int16.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_int16()
_z.resize(1)
vector_int16.elem_type = _z[0].__class__
del(_z)
        
vector_uint16.__old_init__ = vector_uint16.__init__
vector_uint16.__init__ = _c.__vector__init__
vector_uint16.create = _c.__vector_create
vector_uint16.__repr__ = _c.__vector__repr__
vector_uint16.tolist = _c.__vector_tolist
vector_uint16.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_uint16()
_z.resize(1)
vector_uint16.elem_type = _z[0].__class__
del(_z)
        
vector_int.__old_init__ = vector_int.__init__
vector_int.__init__ = _c.__vector__init__
vector_int.create = _c.__vector_create
vector_int.__repr__ = _c.__vector__repr__
vector_int.tolist = _c.__vector_tolist
vector_int.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_int()
_z.resize(1)
vector_int.elem_type = _z[0].__class__
del(_z)
        
vector_uint.__old_init__ = vector_uint.__init__
vector_uint.__init__ = _c.__vector__init__
vector_uint.create = _c.__vector_create
vector_uint.__repr__ = _c.__vector__repr__
vector_uint.tolist = _c.__vector_tolist
vector_uint.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_uint()
_z.resize(1)
vector_uint.elem_type = _z[0].__class__
del(_z)
        
vector_long.__old_init__ = vector_long.__init__
vector_long.__init__ = _c.__vector__init__
vector_long.create = _c.__vector_create
vector_long.__repr__ = _c.__vector__repr__
vector_long.tolist = _c.__vector_tolist
vector_long.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_long()
_z.resize(1)
vector_long.elem_type = _z[0].__class__
del(_z)
        
vector_ulong.__old_init__ = vector_ulong.__init__
vector_ulong.__init__ = _c.__vector__init__
vector_ulong.create = _c.__vector_create
vector_ulong.__repr__ = _c.__vector__repr__
vector_ulong.tolist = _c.__vector_tolist
vector_ulong.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_ulong()
_z.resize(1)
vector_ulong.elem_type = _z[0].__class__
del(_z)
        
vector_int64.__old_init__ = vector_int64.__init__
vector_int64.__init__ = _c.__vector__init__
vector_int64.create = _c.__vector_create
vector_int64.__repr__ = _c.__vector__repr__
vector_int64.tolist = _c.__vector_tolist
vector_int64.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_int64()
_z.resize(1)
vector_int64.elem_type = _z[0].__class__
del(_z)
        
vector_uint64.__old_init__ = vector_uint64.__init__
vector_uint64.__init__ = _c.__vector__init__
vector_uint64.create = _c.__vector_create
vector_uint64.__repr__ = _c.__vector__repr__
vector_uint64.tolist = _c.__vector_tolist
vector_uint64.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_uint64()
_z.resize(1)
vector_uint64.elem_type = _z[0].__class__
del(_z)
        
vector_float32.__old_init__ = vector_float32.__init__
vector_float32.__init__ = _c.__vector__init__
vector_float32.create = _c.__vector_create
vector_float32.__repr__ = _c.__vector__repr__
vector_float32.tolist = _c.__vector_tolist
vector_float32.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_float32()
_z.resize(1)
vector_float32.elem_type = _z[0].__class__
del(_z)
        
vector_float64.__old_init__ = vector_float64.__init__
vector_float64.__init__ = _c.__vector__init__
vector_float64.create = _c.__vector_create
vector_float64.__repr__ = _c.__vector__repr__
vector_float64.tolist = _c.__vector_tolist
vector_float64.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_float64()
_z.resize(1)
vector_float64.elem_type = _z[0].__class__
del(_z)
        
vector_vector_int.__old_init__ = vector_vector_int.__init__
vector_vector_int.__init__ = _c.__vector__init__
vector_vector_int.create = _c.__vector_create
vector_vector_int.__repr__ = _c.__vector__repr__
vector_vector_int.tolist = _c.__vector_tolist
vector_vector_int.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_vector_int()
_z.resize(1)
vector_vector_int.elem_type = _z[0].__class__
del(_z)
        
vector_vector_float32.__old_init__ = vector_vector_float32.__init__
vector_vector_float32.__init__ = _c.__vector__init__
vector_vector_float32.create = _c.__vector_create
vector_vector_float32.__repr__ = _c.__vector__repr__
vector_vector_float32.tolist = _c.__vector_tolist
vector_vector_float32.fromlist = classmethod(_c.__vector_fromlist)
_z = vector_vector_float32()
_z.resize(1)
vector_vector_float32.elem_type = _z[0].__class__
del(_z)
        