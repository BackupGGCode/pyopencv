// dtype - A small module to detect platform attributes and prepare primitive data types
//
// Copyright 2002 Minh-Tri Pham.
// Third party copyrights are property of their respective owners.
//
// This file is part of sdopencv.
//
// This version of dtype is developed from a small part of CrissCross
// written by Steven Noonan, which is available at
// http://code.google.com/p/crisscross/.
//
// sdopencv is licensed under the New BSD License.

/*

Preprocessor Definitions
------------------------

TARGET_CPU_X86
    x86 processor
TARGET_CPU_X64
    64-bit processor
TARGET_CPU_PPC
    PowerPC processor
TARGET_CPU_ALPHA
    Alpha processor (not supported)

TARGET_OS_WINDOWS
    Windows
TARGET_OS_LINUX
    Linux
TARGET_OS_MACOSX
    Mac OS X
TARGET_OS_FREEBSD
    FreeBSD (not supported)
TARGET_OS_NETBSD
    NetBSD (not supported)
TARGET_OS_OPENBSD
    OpenBSD (not supported)

TARGET_COMPILER_GCC
    GNU C++ Compiler
TARGET_COMPILER_VC
    Visual C++
TARGET_COMPILER_ICC
    Intel C++ Compiler

 */

#ifndef __included_cc_platform_detect_h
#define __included_cc_platform_detect_h

#   undef PROCESSOR_DETECTED
#   undef COMPILER_DETECTED
#   undef OS_DETECTED

/* ------------------- *
 * PROCESSOR DETECTION *
 * ------------------- */

/* Carbon defines this for us on Mac, apparently... */
#    if defined ( TARGET_CPU_PPC )
#        define PROCESSOR_DETECTED
#    endif

/* PowerPC */
#    if !defined ( PROCESSOR_DETECTED )
#       if defined ( _ARCH_PPC ) || defined ( __ppc__ ) || defined ( __ppc64__ ) || defined ( __PPC ) || defined ( powerpc ) || defined ( __PPC__ ) || defined ( __powerpc64__ ) || defined ( __powerpc64 )
#           define PROCESSOR_DETECTED
#           if defined ( __ppc64__ ) || defined ( __powerpc64__ ) || defined ( __powerpc64 )
#               define TARGET_CPU_PPC 64
#           else
#               define TARGET_CPU_PPC 32
#           endif
#       endif
#   endif

/* Intel x86 */
#   if !defined ( PROCESSOR_DETECTED )
#       if defined ( __i386__ ) || defined ( __i386 ) || defined ( i386 ) || defined ( _X86_ ) || defined ( _M_IX86 )
#           define PROCESSOR_DETECTED
#           define TARGET_CPU_X86
#       endif
#   endif

/* x64/AMD64 */
#    if !defined ( PROCESSOR_DETECTED )
#       if defined ( __x86_64__ ) || defined ( __x86_64 ) || defined ( __amd64 ) || defined ( __amd64__ ) || defined ( _AMD64_ ) || defined ( _M_X64 )
#           define PROCESSOR_DETECTED
#           define TARGET_CPU_X64
#       endif
#   endif

/* IA64 */
#    if !defined ( PROCESSOR_DETECTED )
#       if defined ( __ia64__ ) || defined ( _IA64 ) || defined ( __ia64 ) || defined ( _M_IA64 )
#           define PROCESSOR_DETECTED
#           define TARGET_CPU_IA64
#       endif
#   endif

/* ------------------- *
 * COMPILER DETECTION  *
 * ------------------- */

#   if !defined ( COMPILER_DETECTED )
#       if defined ( __GNUC__ )
#           define COMPILER_DETECTED
#           define TARGET_COMPILER_GCC
#       endif
#       if defined ( __CYGWIN__ ) || defined ( __CYGWIN32__ )
#           define TARGET_COMPILER_CYGWIN
#       endif
#   endif

#   if !defined ( COMPILER_DETECTED )
#       if defined ( __INTEL_COMPILER ) || defined ( __ICL )
#           define COMPILER_DETECTED
#           define TARGET_COMPILER_ICC
#       endif
#   endif

#   if !defined ( COMPILER_DETECTED )
#       if defined ( _MSC_VER )
#           define COMPILER_DETECTED
#           define TARGET_COMPILER_VC
#       endif
#   endif

/* ------------ *
 * OS DETECTION *
 * ------------ */

#   if !defined ( OS_DETECTED )
#       if defined ( TARGET_COMPILER_VC ) || defined ( _WIN32 ) || defined ( _WIN64 )
#           define OS_DETECTED
#           define TARGET_OS_WINDOWS
#       endif
#   endif

#   if !defined ( OS_DETECTED )
#       if defined ( __linux__ ) || defined ( linux ) || defined ( __linux ) || defined ( __gnu_linux__ ) || defined ( __CYGWIN__ )
#           define OS_DETECTED
#           define TARGET_OS_LINUX
#       endif
#   endif

#   if !defined ( OS_DETECTED )
#       if defined (__FreeBSD__)
#           define OS_DETECTED
#           define TARGET_OS_FREEBSD
#       endif
#   endif

#   if !defined ( OS_DETECTED )
#       if defined (__NetBSD__)
#           define OS_DETECTED
#           define TARGET_OS_NETBSD
#       endif
#   endif

#   if !defined ( OS_DETECTED )
#       if defined (__OpenBSD__)
#           define OS_DETECTED
#           define TARGET_OS_OPENBSD
#       endif
#   endif

#   if !defined ( OS_DETECTED )
#       if defined (__APPLE__) || defined (__MACH__)
#           define OS_DETECTED
#           define TARGET_OS_MACOSX
#       endif
#   endif

#    if defined(TARGET_CPU_IA64) || defined(TARGET_CPU_X64)
#        define TARGET_CPU_BITS 64
#    elif defined(TARGET_CPU_X86)
#        define TARGET_CPU_BITS 32
#    endif

#    if defined(TARGET_CPU_PPC)
#        if TARGET_CPU_PPC == 32
#            define TARGET_CPU_BITS 32
#        elif TARGET_CPU_PPC == 64
#            define TARGET_CPU_BITS 64
#        endif
#    endif

#   if !defined ( PROCESSOR_DETECTED )
#       error "Could not detect target CPU."
#   endif

#   if !defined ( COMPILER_DETECTED )
#       error "Could not detect target compiler."
#   endif

#   if !defined ( OS_DETECTED )
#       error "Could not detect target OS."
#   endif

#   if !defined ( TARGET_CPU_BITS )
#       error "Could not detect 32-bit/64-bit architecture."
#   endif


#   if defined (TARGET_COMPILER_GCC)
#       if defined (__MMX__)
#           define TARGET_COMPILER_SUPPORT_MMX
#       endif
#       if defined (__3dNOW__) || defined (__3dNOW_A__)
#           define TARGET_COMPILER_SUPPORT_3dNOW
#       endif
#       if defined (__SSE__)
#           define TARGET_COMPILER_SUPPORT_SSE
#       endif
#       if defined (__SSE2__)
#           define TARGET_COMPILER_SUPPORT_SSE2
#       endif
#       if defined (__SSE3__)
#           define TARGET_COMPILER_SUPPORT_SSE3
#       endif
#   endif

#   if defined (TARGET_COMPILER_ICC)
#       define TARGET_COMPILER_SUPPORT_MMX
#       define TARGET_COMPILER_SUPPORT_3dNOW
#       define TARGET_COMPILER_SUPPORT_SSE
#       define TARGET_COMPILER_SUPPORT_SSE2
#       define TARGET_COMPILER_SUPPORT_SSE3
#   endif


/* ------------------- *
 * BASIC DATA TYPES  *
 * ------------------- */


#if defined(TARGET_COMPILER_GCC) // assuming C99 compliant
#include <stdint.h>
#endif

namespace sdopencv
{

#if !defined(int8)
#   if defined(TARGET_COMPILER_GCC)
typedef int8_t              int8;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef char                int8;
#   endif
#endif

#if !defined(uint8)
#   if defined(TARGET_COMPILER_GCC)
typedef uint8_t             uint8;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef unsigned char       uint8;
#   endif
#endif

#if !defined(int16)
#   if defined(TARGET_COMPILER_GCC)
typedef int16_t             int16;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef short               int16;
#   endif
#endif

#if !defined(uint16)
#   if defined(TARGET_COMPILER_GCC)
typedef uint16_t            uint16;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef unsigned short      uint16;
#   endif
#endif

#if !defined(int32)
#   if defined(TARGET_COMPILER_GCC)
typedef int32_t             int32;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef int                 int32;
#   endif
#endif

#if !defined(uint32)
#   if defined(TARGET_COMPILER_GCC)
typedef uint32_t            uint32;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef unsigned int        uint32;
#   endif
#endif

#if !defined(int64)
#   if defined(TARGET_COMPILER_GCC)
typedef int64_t             int64;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef __int64             int64;
#   endif
#endif

#if !defined(uint64)
#   if defined(TARGET_COMPILER_GCC)
typedef uint64_t            uint64;
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
typedef unsigned __int64    uint64;
#   endif
#endif

#if !defined(int128)
#   if defined(TARGET_COMPILER_GCC)
#       if defined(int128_t)
typedef int128_t            int128;
#       endif
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
#       if defined(__int128) // x64 compilers
typedef __int128            int128;
#       endif
#   endif
#endif

#if !defined(uint128)
#   if defined(TARGET_COMPILER_GCC)
#       if defined(int128_t)
typedef uint128_t           uint128;
#       endif
#   elif defined(TARGET_COMPILER_ICC) || defined(TARGET_COMPILER_VC)
#       if defined(__int128) // x64 compilers
typedef unsigned __int128   uint128;
#       endif
#   endif
#endif

#if !defined(float32)
typedef float float32;
#endif

#if !defined(float64)
typedef double float64;
#endif



}

#endif

