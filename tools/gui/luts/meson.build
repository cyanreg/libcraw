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

# The embedded film simulation collection.
#
# films/ holds the .cube files exactly as filmgrid.org serves them, which is
# what lives in the repository; see LICENSE.md for where they come from.
# Nothing generated is stored in the tree: the pack is built here, into the
# build directory, and embedded into cineview the way the shaders are.
#
# Hook this in from tools/meson.build with a single
#     subdir('gui/luts')
# placed before the cineview executable, and then add
# cineview_luts_dep to its dependency list.

fs = import('fs')

luts_build = files('../../luts_build.py')
blob_embed = files('../../blob_embed.py')

# zlib is what the pack is compressed with. luts_build.py measures zstd as
# well and only picks it when it wins by a wide margin, which on this data
# it does not (2.9% smaller, and the threshold is 20%), so the reader needs
# nothing beyond zlib. Should that ever flip, add the libzstd dependency and
# -DCINEVIEW_LUTS_HAVE_ZSTD=1 here; luts.c already has the branch.
zlib_dep = dependency('zlib', required: false)

# The collection is optional: a checkout without films/ still builds
# cineview, only without the film simulation menu.
cineview_luts_found = zlib_dep.found() and fs.exists('manifest.json')

if cineview_luts_found
    # One custom_target per step, as with the shaders. luts_build.py writes
    # a depfile naming the 296 .cube files it read, so ninja re-packs when
    # one of them or the manifest changes and skips the step otherwise; a
    # full pack takes about a second, an unchanged one takes none.
    filmgrid_pack = custom_target('filmgrid_pack',
        input: files('manifest.json'),
        output: 'filmgrid.pack',
        depfile: 'filmgrid.pack.d',
        depend_files: luts_build,
        command: [ python_exe, luts_build, 'pack',
                   '--from-dir', meson.current_source_dir(),
                   '--out', '@OUTPUT@', '--depfile', '@DEPFILE@' ],
    )

    filmgrid_pack_h = custom_target('filmgrid_pack_h',
        input: filmgrid_pack,
        output: 'filmgrid_pack.h',
        depend_files: blob_embed,
        command: [ python_exe, blob_embed, '@INPUT@', '@OUTPUT@',
                   'cineview_filmgrid_pack' ],
    )

    # A static library so both cineview and tests/test_luts link the same
    # reader, and so the generated header only has to be wired up once.
    cineview_luts_lib = static_library('cineview_luts',
        files('../luts.c'), filmgrid_pack_h,
        include_directories: [ craw_inc, include_directories('.') ],
        dependencies: [ zlib_dep ],
    )

    cineview_luts_dep = declare_dependency(
        link_with: cineview_luts_lib,
        include_directories: include_directories('..'),
        sources: [ filmgrid_pack_h ],
        dependencies: [ zlib_dep ],
    )
else
    cineview_luts_dep = disabler()
endif

summary({
    'film simulation LUTs': cineview_luts_found,
}, section: 'tools', bool_yn: true)
