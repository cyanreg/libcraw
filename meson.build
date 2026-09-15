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

project('libcraw',
    [ 'c', 'cpp' ],
    license: 'BSD-2-Clause',
    license_files: [ 'LICENSE.md' ],
    default_options: [
        'buildtype=debugoptimized',
        'c_std=c23',
        'cpp_std=c++17',
        'warning_level=1',
    ],
    version: '0.0.1',
    meson_version: '>=1.4.0',
)

# library version
craw_version_split = meson.project_version().split('.')
craw_version_major = craw_version_split[0]
craw_version_minor = craw_version_split[1]
craw_version_micro = craw_version_split[2]

# Python (shader embedding)
#============================================================================
pymod = import('python')
python_exe = pymod.find_installation()

# Compiler
#============================================================================
build_args = [
    '-D_ISOC11_SOURCE',
    '-D_ISOC2X_SOURCE',
    '-D_ISOC23_SOURCE',
    '-U__STRICT_ANSI__',

    # Warnings
    '-Wundef', '-Wmissing-prototypes', '-Wshadow', '-Wparentheses',
    '-Wpointer-arith', '-Wno-pointer-sign',

    # Warnings to treat as errors
    '-Werror=implicit-function-declaration', '-Werror=missing-prototypes',
    '-Werror=vla',
]

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

if cc.has_header('stdckdint.h') == false
    error('Compiler lacks stdckdint.h, cannot compile!')
endif

# Hide all untagged API functions
#============================================================================
if cc.has_argument('-fvisibility=hidden')
    build_args += '-fvisibility=hidden'
else
    warning('Compiler does not support -fvisibility=hidden, all symbols will be public!')
endif

add_project_arguments(build_args, language: 'c')
add_project_arguments(cxx.get_supported_arguments(['-fvisibility=hidden']), language: 'cpp')

# Dependencies
#============================================================================
# Only the headers are needed by the library itself: every entry point is
# loaded through the vkGetInstanceProcAddr the caller hands over. Tests and
# tools create their own device and therefore link the loader.
vulkan_dep = dependency('vulkan')
vulkan_headers_dep = vulkan_dep.partial_dependency(compile_args: true, includes: true)

glslang_prog = find_program('glslangValidator', 'glslang', required: true)

# Configure file
#============================================================================
conf = configuration_data()
conf.set_quoted('PROJECT_VERSION_STRING', meson.project_version())

if get_option('assert') > -1
    conf.set('CONFIG_ASSERT_LEVEL', get_option('assert'))
else
    if get_option('buildtype') == 'release'
        conf.set('CONFIG_ASSERT_LEVEL', 0)
    elif get_option('buildtype') == 'debugoptimized'
        conf.set('CONFIG_ASSERT_LEVEL', 1)
    else
        conf.set('CONFIG_ASSERT_LEVEL', 2)
    endif
endif

# Build
#============================================================================
subdir('include/craw')
subdir('lib')

# The viewer's embedded film simulation collection is built ahead of tests/
# so that tests/test_luts can link the same reader; craw-view picks it up
# from tools/ through craw_view_luts_dep.
if get_option('tools').auto() or get_option('tools').enabled()
    subdir('tools/gui/luts')
endif

if get_option('tests').auto() or get_option('tests').enabled()
    subdir('tests')
endif

if get_option('tools').auto() or get_option('tools').enabled()
    subdir('tools')
endif
