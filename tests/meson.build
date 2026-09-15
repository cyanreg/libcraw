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
