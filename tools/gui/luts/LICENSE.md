# Film simulation LUTs — provenance and attribution

The `.cube` files under `films/` are **not** part of libcraw and are **not**
covered by libcraw's BSD-2-Clause licence. They were fetched from
<https://filmgrid.org/> on 2026-09-15 and are stored here byte for byte as
that site serves them. Everything else in this directory — `meson.build`,
`manifest.json`, and the pack format built from these files — is libcraw's
own work under BSD-2-Clause.

This file records what was found about where the data comes from. It is a
record, not legal advice.

## What the files themselves say

Every one of the 296 `.cube` files opens with the line

    # Created by: G'MIC (https://gmic.eu)

so the collection's provenance is stated in the data itself: these are the
G'MIC film emulation CLUTs, exported to the Adobe Cube LUT format.

## What filmgrid.org states

The footer of every page on <https://filmgrid.org/> carries, verbatim:

> Film stock names are trademarks of their respective owners. Filmgrid is not
> affiliated with, endorsed by or sponsored by Eastman Kodak Company, FUJIFILM
> Corporation, Harman Technology (Ilford), Polaroid, Agfa-Gevaert or any other
> manufacturer. Stock names are used descriptively to identify the film each
> look emulates. The underlying LUTs derive from the open-source G'MIC film
> emulation collection and are provided as-is for creative use.

The site's `robots.txt` (<https://filmgrid.org/robots.txt>) is
`User-agent: * / Allow: /` and was honoured; the crawl ran one request at a
time with a delay and an identifying User-Agent.

The film stock names carried in `manifest.json` are trademarks of their
respective owners and are used here, as on the source site, descriptively —
to say which film each rendering emulates. libcraw is not affiliated with,
endorsed by or sponsored by any film manufacturer.

## Upstream: the G'MIC film emulation collection

The collection originates with **Pat David**, who built the film emulation
CLUTs that ship with G'MIC and that RawTherapee bundles as its Film
Simulation collection.

- Pat David, *Film Emulation in RawTherapee*:
  <https://patdavid.net/2015/03/film-emulation-in-rawtherapee/>
- Pat David, *Film Emulation Presets in G'MIC (GIMP)*:
  <https://patdavid.net/2013/08/film-emulation-presets-in-gmic-gimp/>
- G'MIC colour presets: <https://gmic.eu/color_presets/>
- RawPedia, *Film Simulation*:
  <https://rawpedia.rawtherapee.com/Film_Simulation>

patdavid.net carries a site-wide licence notice, verbatim:

> Unless otherwise noted, this work by Pat David is licensed under a Creative
> Commons Attribution-ShareAlike 4.0 International License.

RawPedia's Film Simulation page credits the RawTherapee Film Simulation
collection to **Pat David, Pavlov Dmitry and Michael Ezra**, and states no
separate licence for the CLUT files beyond that credit. G'MIC itself — the
program that generated these files — is distributed under the CeCILL
licences (<https://gmic.eu/>).

## Attribution

Carry this credit wherever the collection is redistributed or displayed:

> Film simulation LUTs from the G'MIC film emulation collection by Pat David
> (CC BY-SA 4.0, <https://creativecommons.org/licenses/by-sa/4.0/>), with
> contributions credited by RawTherapee to Pavlov Dmitry and Michael Ezra,
> obtained via <https://filmgrid.org/>.

## Caveats, recorded rather than resolved

- No licence file is served alongside the `.cube` files themselves; the
  CC BY-SA 4.0 above is patdavid.net's site-wide notice, which is the
  closest thing to an explicit grant that could be found for this data.
- CC BY-SA is a **share-alike** licence. It asks that adaptations of the
  licensed material be offered under the same terms. The LUT data here is
  carried unmodified and kept separately licensed from libcraw's own
  BSD-2-Clause code for exactly that reason; whether the pack format built
  from it counts as an adaptation was not adjudicated.
- Filmgrid's own wording, "provided as-is for creative use", is a
  permission to use rather than a formal licence grant.

The project owner reviewed the above and decided to proceed.

## Reproducing this directory

    tools/luts_build.py fetch --dir tools/gui/luts

re-crawls the site and rewrites `films/` and `manifest.json`. The build only
ever runs `tools/luts_build.py pack`, which never touches the network.
