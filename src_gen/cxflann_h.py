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
# cxflann.h
#=============================================================================


    ''')
       
    # Index: there are two classes, one from namespace 'flann', the other from namespace 'cv::flann'
    flanns = mb.classes('Index')
    flanns.include()
    if flanns[0].decl_string == '::flann::Index':
        flann_Index = flanns[0]
        cvflann_Index = flanns[1]
    else:
        flann_Index = flanns[1]
        cvflann_Index = flanns[0]
    flann_Index.rename('flann_Index')
    
    mb.init_class(cvflann_Index)
    for t in ('knnSearch', 'radiusSearch'):
        for z in cvflann_Index.mem_funs(t):
            z._transformer_kwds['alias'] = t
        z = cvflann_Index.mem_fun(lambda x: x.name==t and 'vector' in x.decl_string)
        z._transformer_creators.append(FT.output_std_vector('indices'))
        z._transformer_creators.append(FT.output_std_vector('dists'))
    mb.finalize_class(cvflann_Index)
    
    # IndexParams
    mb.class_('IndexParams').include()
    
    # IndexFactory classes
    for name in (
        'IndexFactory',
        'LinearIndexParams', 'KDTreeIndexParams', 'KMeansIndexParams',
        'CompositeIndexParams', 'AutotunedIndexParams', 'SavedIndexParams', 
        ):
        z = mb.class_(name)
        mb.init_class(z)
        FT.expose_func(z.mem_fun('createIndex'))
        mb.finalize_class(z)

    # SearchParams
    mb.class_('SearchParams').include()

    mb.free_fun('hierarchicalClustering').include()

