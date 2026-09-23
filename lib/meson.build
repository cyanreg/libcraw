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
subdir('shaders')

craw_sources = files(
    # Context, logging, version
    'craw.c',

    # Vulkan runtime
    'vk_functions.c',
    'vk_buffer.c',
    'vk_image.c',
    'vk_shader.c',
    'vma/vma.cpp',

    # Colour science (CPU)
    'profile.c',
    'colorspace.c',
    'gamut.c',
    'awb.c',
    'awb_smoother.c',
    'noise.c',

    # Vendor lens correction records (CPU, no Vulkan)
    'lens.c',

    # GPU pipeline
    'pipeline.c',

    # CPU reference (tests and tools only, not exported)
    'ref/ref_debayer.c',
    'ref/ref_color.c',
    'ref/ref_denoise.c',
    'ref/ref_false_color.c',
    'ref/ref_chroma.c',
    'ref/ref_lens.c',
    'ref/ref_highlight.c',
)

# Generate local configure file
configure_file(
    output: 'config.h',
    configuration: conf,
)

craw_lib_inc = include_directories('.')

if host_machine.system() == 'windows'
    craw_soversion = ''
else
    craw_soversion = meson.project_version()
endif

craw_both = both_libraries('craw',
    craw_sources,
    craw_shader_headers,
    install: true,
    include_directories: [ craw_inc, craw_lib_inc, craw_shader_inc ],
    dependencies: [ vulkan_headers_dep ],
    version: meson.project_version(),
    soversion: craw_soversion,
    cpp_args: [ '-DVMA_STATIC_VULKAN_FUNCTIONS=0', '-DVMA_DYNAMIC_VULKAN_FUNCTIONS=1' ],
)
craw_lib = craw_both.get_shared_lib()

craw_dep = declare_dependency(
    link_with: craw_lib,
    include_directories: craw_inc,
)

# Tests link the static library so they can reach the CPU reference.
craw_static_dep = declare_dependency(
    link_with: craw_both.get_static_lib(),
    include_directories: [ craw_inc, craw_lib_inc ],
)

meson.override_dependency('libcraw', craw_dep)

pkg = import('pkgconfig')
pkg.generate(
    name: meson.project_name(),
    description: 'Cinema RAW processing library',
    libraries: craw_lib,
    version: meson.project_version(),
)

summary({
    'glslang': glslang_prog.found(),
}, section: 'libcraw', bool_yn: true)
