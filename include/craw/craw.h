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

/*
 * libcraw: Cinema RAW processing library.
 *
 * Turns camera-native Bayer sensor data into scene-linear colour on the GPU,
 * using Vulkan compute exclusively. The first supported source is Panasonic
 * ProRes RAW as decoded by FFmpeg; ARRIRAW and BRAW are planned. The library
 * never parses container or bitstream data: levels, white balance, matrices
 * and CCT tables arrive through a CRAWProfile (<craw/profile.h>) that the
 * caller fills from whatever the decoder exposed.
 *
 * Ownership model: libcraw never creates a VkInstance, VkDevice,
 * VkCommandPool, VkCommandBuffer, or submits anything to a queue. The caller
 * owns all of that. The caller:
 *
 *   1. Creates a VkInstance/VkDevice with the extensions/features libcraw
 *      requires (see craw_vulkan_get_requirements() in <craw/vulkan.h>).
 *   2. Calls craw_context_create() once, handing over the raw Vulkan handles.
 *   3. Fills a CRAWProfile describing the camera and the clip.
 *   4. Creates a CRAWPipeline bound to a frame size and a stage configuration
 *      via craw_pipeline_create() (<craw/pipeline.h>).
 *   5. Per frame: begins its own VkCommandBuffer, calls
 *      craw_pipeline_record() to have libcraw record dispatches + barriers
 *      into it, ends and submits the command buffer itself.
 *   6. Optionally reads back the auto white balance histogram and runs the
 *      CPU-side estimator and temporal smoother from <craw/awb.h>.
 *
 * Internal allocations (intermediate images, histogram buffers) are made on
 * the caller's VkDevice through the AMD Vulkan Memory Allocator.
 *
 * libcraw is not thread-safe across a single CRAWContext/CRAWPipeline
 * without external locking; it performs no internal synchronization.
 */

#ifndef CRAW_H
#define CRAW_H

#include <stdint.h>
#include <stdarg.h>

#include <craw/utils.h>
#include <craw/version.h>

typedef struct CRAWContext CRAWContext;

enum CRAWLogLevel {
    CRAW_LOG_QUIET    = -(1 << 0),
    CRAW_LOG_FATAL    =  (0 << 0),
    CRAW_LOG_ERROR    = +(1 << 0),
    CRAW_LOG_WARN     = +(1 << 1),
    CRAW_LOG_INFO     = +(1 << 2),
    CRAW_LOG_VERBOSE  = +(1 << 3),
    CRAW_LOG_DEBUG    = +(1 << 4),
    CRAW_LOG_TRACE    = +(1 << 5),
};

/* Library context-level options. All fields are optional (zero-initialize
 * to get defaults). */
typedef struct CRAWContextOptions {
    /* Logging context */
    void *log_opaque;

    /* Logging callback. If NULL, messages at CRAW_LOG_WARN and above go to
     * stderr. */
    void (*log_cb)(void *log_opaque, enum CRAWLogLevel level,
                   const char *format, va_list args);

    /* Lowest level that is passed to log_cb. */
    enum CRAWLogLevel log_level;

    /* Padding to allow for future options. Must always be set to 0. */
    uint8_t padding[1024 - 1*4 - 2*8];
} CRAWContextOptions;

/* Releases a CRAWContext and everything it owns (pipelines must already be
 * freed). Does not touch the caller's VkDevice. */
CRAW_API void craw_context_free(CRAWContext **ctx);

/* Main logging function */
CRAW_API void craw_log(void *ctx, enum CRAWLogLevel level, const char *fmt, ...) craw_printf_format(3, 4);

#endif /* CRAW_H */
