import distutils.cygwinccompiler as dcyg

# the same as cygwin plus some additional parameters -- modified by Minh-Tri Pham
class Mingw32CCompiler (dcyg.CygwinCCompiler):

    compiler_type = 'mingw32'

    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):

        dcyg.CygwinCCompiler.__init__ (self, verbose, dry_run, force)

        # ld_version >= "2.13" support -shared so use it instead of
        # -mdll -static
        if self.ld_version >= "2.13":
            shared_option = "-shared"
        else:
            shared_option = "-mdll -static"

        # A real mingw32 doesn't need to specify a different entry point,
        # but cygwin 2.91.57 in no-cygwin-mode needs it.
        if self.gcc_version <= "2.91.57":
            entry_point = '--entry _DllMain@12'
        else:
            entry_point = ''

        self.set_executables(compiler='gcc -mno-cygwin -O -Wall',
                             compiler_so='gcc -mno-cygwin -O -Wall', # here, -mdll is removed because with -mdll, Python extensions via boost python fail
                             compiler_cxx='g++ -mno-cygwin -O -Wall',
                             linker_exe='gcc -mno-cygwin',
                             linker_so='%s -mno-cygwin %s %s'
                                        % (self.linker_dll, shared_option,
                                           entry_point))
        # Maybe we should also append -mthreads, but then the finished
        # dlls need another dll (mingwm10.dll see Mingw32 docs)
        # (-mthreads: Support thread-safe exception handling on `Mingw32')

		# no additional libraries needed
        self.dll_libraries=[]

        # CMake says don't use msvcr90 but instead use this set of dll libraries
        self.dll_libraries=['kernel32', 'user32', 'gdi32', 'winspool', 'shell32', 'ole32', 'oleaut32', 'uuid', 'comdlg32', 'advapi32']
 
        # Include the appropriate MSVC runtime library if Python was built
        # with MSVC 7.0 or later.
        # self.dll_libraries = dcyg.get_msvcr()

    # __init__ ()

# hack
dcyg.Mingw32CCompiler = Mingw32CCompiler
