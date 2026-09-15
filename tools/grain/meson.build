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

# grain_preview: the headless proof for the film grain model.
#
# It makes a libplacebo Vulkan context with no surface, renders a test chart
# and a developed frame through exactly the pl_render_image path craw-view
# uses, applies grain by each path and each preset, measures what came out
# and writes raw frames and contact sheets. Every number in DESIGN.md's "Film grain
# emulation" section came out of this, including the calibration tables
# grain.c carries: `grain_preview --calibrate` re-measures them against
# whatever libplacebo is installed.
#
# Not installed, and not a `test()`: it needs a GPU and it writes pictures a
# person has to look at. Hooked in from tools/meson.build with one line,
#
#     subdir('grain')
#
# inside the `if craw_tools_gui` block.

executable('grain_preview',
    'grain_preview.c',
    '../gui/grain.c',
    c_args: gui_args,
    include_directories: [ include_directories('../gui') ],
    dependencies: [
        placebo,
        libavformat_dep, libavcodec_dep, libavutil_dep,
        cc.find_library('m', required: false),
    ],
    build_by_default: false,
)
