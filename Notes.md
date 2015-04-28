# Notes to remind me how I should develop PyOpenCv #

## Crash issue with CV\_SSE2 ##
During the compile time of OpenCV 2.0, if CV\_SSE2 is enabled, assembly SSE2 code is used instead of C code. This works fine in C/C++ but **crashes** in Python. Therefore, disabling SSE2 when compiling OpenCV 2.0 is a must.

_Update:_ in fact, if any of the following GCC options is enabled, my OpenCV 2.0 build for Windows crashes when it is being run in Python: -msse2, -mfpmath=sse, -march=pentium3, -march=pentium4