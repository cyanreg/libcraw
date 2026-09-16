# Copyright © 2026, Lynne
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# CLI tools. cineraw decodes a camera clip with FFmpeg and runs every
# frame through the library on the GPU, so it needs all three FFmpeg
# libraries; without them there is nothing here to build.
libavformat_dep = dependency('libavformat', required: get_option('ffmpeg'))
libavcodec_dep = dependency('libavcodec', required: get_option('ffmpeg'))
libavutil_dep = dependency('libavutil', required: get_option('ffmpeg'))

# cineview additionally needs a window, a renderer and a UI.
sdl3 = dependency('sdl3', required: false)
placebo = dependency('libplacebo', version: '>=7.349', required: false)
# Audio is optional: without libswresample cineview plays silent and paces
# the picture off wall time instead of the audio clock.
swresample = dependency('libswresample', required: false)
# The still exporter converts the rendered RGBA16 into whatever the encoder
# the file name picks will take, which for a JPEG or an AVIF is YUV.
swscale = dependency('libswscale', required: false)
# The clip metadata a camera writes into the container is an XML document -
# Panasonic's, and ARRIRAW's is another - and the values in it decide how
# the picture is read. Hand-matching elements in a blob is the kind of thing
# that works until a camera indents differently, so it is parsed properly.
libxml = dependency('libxml-2.0', required: false)

craw_tools_ffmpeg = libavformat_dep.found() and libavcodec_dep.found() and libavutil_dep.found()
craw_tools_gui = craw_tools_ffmpeg and sdl3.found() and placebo.found() and \
                 swscale.found() and libxml.found()

if craw_tools_ffmpeg
    tools_args = []

    # AVRawColorParams grew the exposure index fields after the first
    # release that had the struct at all; the profile simply leaves them at
    # zero on an older libavutil, which craw_profile_eval() reads as
    # "unknown".
    if cc.has_member('AVRawColorParams', 'as_shot_ei',
                     prefix: '#include <libavutil/raw_color_params.h>',
                     dependencies: libavutil_dep)
        tools_args += '-DCRAW_TOOL_HAVE_EXPOSURE_INDEX=1'
    else
        tools_args += '-DCRAW_TOOL_HAVE_EXPOSURE_INDEX=0'
    endif

    gui_args = tools_args
    gui_deps = []
    if swresample.found()
        gui_args += '-DCINEVIEW_HAVE_AUDIO=1'
        gui_deps += swresample
    else
        gui_args += '-DCINEVIEW_HAVE_AUDIO=0'
    endif

    # The embedded film simulation collection, built by the top level's
    # subdir('tools/gui/luts') ahead of tests/ so that tests/test_luts can
    # link the same reader. Only the dependency is added here, and only
    # when the collection is really there: cineview_luts_dep is a disabler
    # otherwise, and a disabler anywhere in an executable's dependency list
    # disables the whole target, which would make cineview vanish from a
    # checkout without the .cube files instead of merely building without
    # the menu.
    if cineview_luts_found
        gui_args += '-DCINEVIEW_HAVE_LUTS=1'
        gui_deps += cineview_luts_dep
    else
        gui_args += '-DCINEVIEW_HAVE_LUTS=0'
    endif

    executable('cineraw',
        'cineraw.c',
        'profiles.c',
        # The test device helper doubles as the tool's Vulkan boilerplate.
        '../tests/vkdev.c',
        c_args: tools_args,
        include_directories: [ craw_inc, include_directories('../tests') ],
        dependencies: [
            craw_dep, vulkan_dep,
            libavformat_dep, libavcodec_dep, libavutil_dep,
            cc.find_library('m', required: false),
        ],
        install: true,
    )

    # cineview: the interactive viewer and exporter. libplacebo owns the
    # VkDevice, FFmpeg's Vulkan hwaccel decodes onto it, libcraw processes
    # in place and nuklear draws the panel into an overlay image.
    if craw_tools_gui
        executable('cineview',
            'gui/main.c',
            'gui/gpu.c',
            'gui/decode.c',
            'gui/ui.c',
            'gui/audio.c',
            'gui/export.c',
            'gui/still.c',
            'gui/encode.c',
            'gui/grain.c',
            'gui/grain_ui.c',
            'gui/adjust.c',
            'profiles.c',
            c_args: gui_args,
            include_directories: [ craw_inc ],
            dependencies: [
                craw_dep, vulkan_dep, sdl3, placebo,
                libavformat_dep, libavcodec_dep, libavutil_dep, swscale,
                libxml,
                cc.find_library('m', required: false),
            ] + gui_deps,
            install: true,
        )

        # The film grain model's headless proof. Not installed and not a
        # test(): it needs a GPU and it writes pictures a person looks at.
        subdir('grain')
    endif
endif

summary({
    'cineraw': craw_tools_ffmpeg,
    'cineview': craw_tools_gui,
    'cineview audio': craw_tools_gui and swresample.found(),
}, section: 'tools', bool_yn: true)
