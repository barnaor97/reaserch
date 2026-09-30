# Restricted inputs - place your authorized copies here

This directory is intentionally empty and is git-ignored. It is kept as a
reminder that this repository is not self-contained.

The thesis depends on two third-party research datasets that are **not
redistributed** here. Obtain authorized copies and place them at the paths the
scripts open:

    data/thesis_maps/fig_gt_fires.shp   (+ .shx .dbf .prj .cpg)   fire-reference inventory
    data/raw/landcover_2018_30m.tif                               land-cover product

Both paths are git-ignored, so a stray copy cannot be committed by accident.

Required formats, CRS, schemas, class codes and the exact role each dataset plays
in the workflow are documented in ../README.md (sections 2.1 and 2.2).

Do not commit any restricted data anywhere in this repository.
