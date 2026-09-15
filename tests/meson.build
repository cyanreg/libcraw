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
# Test/tool device helper: creates a Vulkan 1.4 instance and device with
# everything craw_vulkan_get_requirements() asks for. Links the loader.
craw_testdev = static_library('craw_testdev',
    'vkdev.c',
    include_directories: [ craw_inc ],
    dependencies: [ vulkan_dep ],
)
craw_testdev_dep = declare_dependency(
    link_with: craw_testdev,
    include_directories: include_directories('.'),
    dependencies: [ vulkan_dep ],
)

# CPU-only tests
foreach t : [ 'test_profile', 'test_ref_debayer', 'test_awb' ]
    exe = executable(t, t + '.c', dependencies: [ craw_static_dep ], link_language: 'cpp')
    test(t, exe, timeout: 120)
endforeach

# GPU tests: exit with 77 (skipped) when no suitable device exists.
foreach t : [ 'test_pipeline' ]
    exe = executable(t, t + '.c',
        dependencies: [ craw_static_dep, craw_testdev_dep ],
        link_language: 'cpp',
        c_args: [ '-DTEST_SOURCE_DIR="' + meson.current_source_dir() + '"' ])
    test(t, exe, timeout: 300)
endforeach

# The embedded film simulation collection. Its pack is built by
# tools/gui/luts/meson.build; this test only exists once that has been
# processed, which — because meson walks subdir() in order and tests/ comes
# before tools/ — means the hook line has to run before subdir('tests'):
#
#     subdir('tools/gui/luts')   in the top level meson.build, or
#     subdir('gui/luts')         in tools/meson.build, with tests moved after
#
# Until then this is simply skipped, so a checkout without the hook, without
# the .cube files or without zlib still builds and tests everything else.
if is_variable('cineview_luts_dep')
    exe = executable('test_luts', 'test_luts.c',
        dependencies: [ cineview_luts_dep, cc.find_library('m', required: false) ],
        c_args: [ '-DTEST_LUTS_DIR="' +
                  meson.project_source_root() / 'tools' / 'gui' / 'luts' + '"' ])
    test('test_luts', exe, timeout: 300)
endif

# The film grain parameter model. It lives in tools/gui/ because that is
# where cineview uses it from, but it has no SDL and no nuklear in it, so
# the test builds it directly. libplacebo is asked for again here rather
# than reused from tools/, because meson walks tests/ first.
grain_placebo = dependency('libplacebo', version: '>=7.349', required: false)
if grain_placebo.found()
    exe = executable('test_grain',
        'test_grain.c',
        '../tools/gui/grain.c',
        include_directories: include_directories('../tools/gui'),
        dependencies: [ grain_placebo, cc.find_library('m', required: false) ])
    test('test_grain', exe, timeout: 300)
endif
