import pyopencv as cv

ofs = open('pyopencv_epydoc.cfg','wt')

ofs.write('''
[epydoc] # Epydoc section marker (required by ConfigParser)

# Information about the project.
name: PyOpenCV
url: http://code.google.com/p/pyopencv/

# The list of modules to document.  Modules can be named using
# dotted names, module filenames, or package directory names.
# This option may be repeated.
modules: pyopencv''')

for decl in cv.__dict__.keys():
    if decl.startswith('_'):
        continue
    class_name = eval('cv.'+decl+'.__class__.__name__')
    if not (class_name in ['class', 'function']):
        continue
    ofs.write(', pyopencv.'+decl)

ofs.write('''

# Write html output to the directory "apidocs"
output: html
target: apidocs/

# Include all automatically generated graphs.  These graphs are
# generated using Graphviz dot.
# graph: all
# dotpath: /usr/local/bin/dot
''')
