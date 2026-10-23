# Data

## Source

All soil data are derived from the **WoSIS snapshot – December 2023** (ISRIC – World Soil Information):

- Dataset: Calisto, L., de Sousa, L. M., Batjes, N. H. (2023). *Standardised soil profile data for the
  world (WoSIS snapshot – December 2023)*. ISRIC Data Hub. https://doi.org/10.17027/isric-wdcsoils-20231130
- Catalogue entry and download:
  https://data.isric.org/srv/eng/catalog.search#/metadata/e50f84e1-aa5b-49cb-bd6b-cd581232a2ec
- Methodology: Batjes, N. H., Calisto, L., de Sousa, L. M. (2024). Providing quality-assessed and
  standardised soil data to support global mapping and modelling (WoSIS snapshot 2023).
  *Earth System Science Data*, 16, 4735–4765. https://doi.org/10.5194/essd-16-4735-2024

**Licence.** The data provider sets the licence of each WoSIS profile (CC-BY or CC-BY-NC). The derived
files below carry the same per-profile terms. Profiles under CC-BY-NC may not be used commercially.
The MIT licence of this repository applies to the code only.

The environmental covariates attached to each profile come from:

- Climate (monthly mean near-surface air temperature and precipitation): World Bank Climate Change
  Knowledge Portal, ERA5-derived gridded data.
- Land cover: MODIS MCD12Q1 Land Cover Type, IGBP classification (`landcover`).
- Ecoregions: CONABIO biotic regionalisation of Mexico (`zone_number`, `zone_name`, `ecoregion_type`).

Section 3.2 of the paper gives the full references.

## Files in `processed/`

| File | Rows | Profiles | Target |
|---|---|---|---|
| `Mexico_standardized_cfvo_orgc.csv` | 10,016 | 3,326 | `orgc`: organic carbon (g/kg) |
| `Mexico_standardized_cfvo_tceq.csv` | 5,635 | 1,872 | `tceq`: calcium carbonate equivalent, total (g/kg) |

Each row is one harmonised depth layer (`depth_category` = `0_5`, `5_15` or `15_30` cm) of one soil
profile (`profile_id`). Rows with missing values are removed by the modelling scripts, which leaves
6,024 rows / 2,002 profiles (ORGC) and 3,898 rows / 1,297 profiles (TCEQ).

| Column | Description | Unit |
|---|---|---|
| `profile_id` | WoSIS profile identifier | – |
| `depth_category` | harmonised layer | cm |
| `date` | sampling date | – |
| `longitude`, `latitude` | profile location (WGS84) | degrees |
| `clay`, `sand`, `silt` | particle-size fractions | % |
| `elcosp` | electrical conductivity of saturated paste (ORGC file only) | dS/m |
| `phaq` | pH in water | – |
| `orgc` / `tceq` | target property | g/kg |
| `soil_type` | soil classification (FAO 1974 legend, Spanish names) | – |
| `slopemean` | mean terrain slope at the profile location | as in source layer |
| `bedrock` | bedrock type | – |
| `cfr` | coarse fragments | % volume |
| `temperature`, `precipitation` | monthly mean climate at the profile location | °C, mm |
| `ecoregion_type`, `zone_number`, `zone_name` | CONABIO biotic region | – |
| `landcover` | MODIS IGBP land-cover class | – |

## Regenerating the processed files

1. Download the WoSIS snapshot from the link above and extract the layer and profile tables for Mexico.
2. Harmonise depths with `r/ea_spline_generator.R` (see `docs/TUTORIAL.md`, step 4).
3. Attach the climate, land-cover and ecoregion covariates by spatial join on the profile coordinates.
   The original scripts for this step are in
   [tierrafoods-hud/notebooks](https://github.com/tierrafoods-hud/notebooks) (`wosis-preprocessor.ipynb`,
   `soil-depth-standardization.ipynb`) and
   [tierrafoods-hud/nasa_landcover](https://github.com/tierrafoods-hud/nasa_landcover).
