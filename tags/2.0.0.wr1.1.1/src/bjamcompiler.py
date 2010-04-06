"""A simple bjam compiler that builds static Python extensions using gcc"""

import os, sys
from distutils import ccompiler

def mypath(path):
    return path.replace('\\', '\\\\')

boost_dir = None

class BjamCompiler(ccompiler.CCompiler):
    compiler_type = 'bjam'

    executables = {}

    def compile(self, sources, **kwds):
        return (sources, kwds)
        
    def link (self,
              target_desc,
              objects,
              output_filename,
              output_dir=None,
              libraries=None,
              library_dirs=None,
              runtime_library_dirs=None,
              export_symbols=None,
              debug=0,
              extra_preargs=None,
              extra_postargs=None,
              build_temp=None,
              target_lang=None):
        # generate file 'Jamroot'
        f = open('Jamroot', 'wt')
        f.write('''
import python ;

using gcc :  :  g++ : <compileflags>-O3
''')
        for include_dir in objects[1]['include_dirs']:
            f.write('    <compileflags>-I%s\n' % mypath(include_dir))
        for library in libraries:
            f.write('    <linkflags>-l%s\n' % library)
        for library_dir in library_dirs:
            f.write('    <linkflags>-L%s\n' % mypath(library_dir))
        f.write(''' ;
if ! [ python.configured ]
{
    ECHO "notice: no Python configured in user-config.jam" ;
    ECHO "notice: will use default configuration" ;
    using python ;
}

use-project boost
  : %s ;

project
  : requirements <library>/boost/python//boost_python ;

python-extension pyopencvext :
''' % (boost_dir, ))
        for source in objects[0]:
            f.write('    %s\n' % mypath(source))
        f.write(''' : ;

import testing ;

testing.make-test run-pyd : pyopencvext _get_ext.py : : test_ext ;
        
alias test : test_ext ;

explicit test_ext test_embed test ;
''')
        f.close()
        
        # boost-build.jam
        f = open('boost-build.jam', 'wt')
        f.write('''
boost-build %s/tools/build/v2 ;
''' % (boost_dir, ))
        
        # script to get the extension
        f = open('_get_ext.py', 'wt')
        f.write('''
import pyopencvext as _P
import distutils.file_util as _D
_D.copy_file(_P.__file__, '%s')
''' % mypath(os.path.abspath(output_filename)))
        f.close()
        
        # build the extension
        self.spawn(['bjam', 'release', 'test', 'link=static'])
        
        # delete temp files
        os.remove('boost-build.jam')
        os.remove('Jamroot')
        os.remove('_get_ext.py')
        

ccompiler.BjamCompiler = BjamCompiler
ccompiler.compiler_class['bjam'] = ('ccompiler', 'BjamCompiler', "A simple bjam compiler that builds static Python extensions using gcc")
