# Modifications to GRADE

This directory contains a modified copy of GRADE, originally by
Farbod Mahmoudinobar and Cristiano L. Dias.

- Upstream: https://github.com/farbod-nobar/GRADE
- Upstream paper: Mahmoudinobar & Dias, *Comput. Phys. Commun.* **244**, 385 (2019)
- Upstream license: GPL-3.0-or-later (see `LICENSE` in this directory)

## What we changed

The upstream version detects the `5^12`, `5^12 6^2`, and `5^12 6^4` cages.
For the CO₂ clathrate hydrate study in this repository, we additionally
need to identify the `4^1 5^10 6^2` cage, which is characteristic of the
amorphous nucleation channel (see the Supplementary Material, Sec. S-II.C).

Specifically, we:

- Added recognition of `4^1 5^4 6^2` cups and their combination with `5^6`
  cups to build `4^1 5^10 6^2` cages.
- Extended the cup-matching logic in `GRADE.cpp` accordingly.
- Added an output file `cage4151062_grade.txt` listing the per-frame count
  of `4^1 5^10 6^2` cages, analogous to the existing `5^12 6^2` output.

## License

Because this is a modified version of a GPL-3.0-or-later work, this
directory is distributed under the same license. See `LICENSE`.

The rest of the repository (`src/`, `analysis_scripts/`, `notebooks/`)
calls GRADE as an external subprocess and is distributed under MIT;
see the top-level `LICENSE`.