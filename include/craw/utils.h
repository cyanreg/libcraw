/*
 * Copyright © 2026, Lynne
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CRAW_UTILS_H
#define CRAW_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

/* All functions return negative values for errors.
 * This is a wrapper that's used to convert standard errno values into
 * negatives. */
#define CRAW_ERROR(err) (-(err))

#define CRAW_MIN(v1, v2) ((v1) < (v2) ? (v1) : (v2))
#define CRAW_MAX(v1, v2) ((v1) > (v2) ? (v1) : (v2))
#define CRAW_CLAMP(v, lo, hi) CRAW_MIN(CRAW_MAX((v), (lo)), (hi))
#define CRAW_ARRAY_ELEMS(arr) (sizeof(arr) / sizeof((arr)[0]))

/* Any function flagged with CRAW_API is exported as a symbol */
#ifndef CRAW_API
    #if defined _WIN32
        #if defined CRAW_BUILDING_DLL
            #define CRAW_API __declspec(dllexport)
        #else
            #define CRAW_API
        #endif
    #else
        #if __GNUC__ >= 4
            #define CRAW_API __attribute__ ((visibility ("default")))
        #else
            #define CRAW_API
        #endif
    #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define craw_printf_format(fmtpos, attrpos) __attribute__((__format__(__printf__, fmtpos, attrpos)))
#else
#define craw_printf_format(fmtpos, attrpos)
#endif

#endif /* CRAW_UTILS_H */
