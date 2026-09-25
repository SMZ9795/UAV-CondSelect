# Configuration index

Run configurations through [run.py](../tools/run.py). The [experiment guide](../docs/reproduction.md) maps paper tables to commands and results; [protocols](../docs/protocols.md) describe training and evaluation.

## Main detection experiments

R, S, F, D, and L denote rain, snow, fog, dust, and low light. All subsets retain clean images. Links below use seed0; matching seed1 and seed2 files are in the same directory. Rows follow subset size and R/S/F/D/L combination order.

| Subset | RT internal | RT external | YOLO external |
| --- | --- | --- | --- |
| Clean-only | [config](rt/internal_clean_only_seed0.json) | [config](rt/external_clean_only_seed0.json) | [config](yolo/s01_external_seed0.json) |
| R | [config](rt/internal_rain_only_seed0.json) | [config](rt/external_rain_only_seed0.json) | [config](yolo/s28_external_seed0.json) |
| S | [config](rt/internal_snow_only_seed0.json) | [config](rt/external_snow_only_seed0.json) | [config](yolo/s29_external_seed0.json) |
| F | [config](rt/internal_fog_only_seed0.json) | [config](rt/external_fog_only_seed0.json) | [config](yolo/s32_external_seed0.json) |
| D | [config](rt/internal_dust_only_seed0.json) | [config](rt/external_dust_only_seed0.json) | [config](yolo/s30_external_seed0.json) |
| L | [config](rt/internal_dim_low_light_seed0.json) | [config](rt/external_dim_low_light_seed0.json) | [config](yolo/s31_external_seed0.json) |
| RS | [config](rt/internal_rain_snow_seed0.json) | [config](rt/external_rain_snow_seed0.json) | [config](yolo/s18_external_seed0.json) |
| RF | [config](rt/internal_rain_fog_seed0.json) | [config](rt/external_rain_fog_seed0.json) | [config](yolo/s21_external_seed0.json) |
| RD | [config](rt/internal_rain_dust_seed0.json) | [config](rt/external_rain_dust_seed0.json) | [config](yolo/s19_external_seed0.json) |
| RL | [config](rt/internal_rain_dim_seed0.json) | [config](rt/external_rain_dim_seed0.json) | [config](yolo/s20_external_seed0.json) |
| SF | [config](rt/internal_snow_fog_seed0.json) | [config](rt/external_snow_fog_seed0.json) | [config](yolo/s24_external_seed0.json) |
| SD | [config](rt/internal_snow_dust_seed0.json) | [config](rt/external_snow_dust_seed0.json) | [config](yolo/s22_external_seed0.json) |
| SL | [config](rt/internal_snow_dim_seed0.json) | [config](rt/external_snow_dim_seed0.json) | [config](yolo/s23_external_seed0.json) |
| FD | [config](rt/internal_fog_dust_seed0.json) | [config](rt/external_fog_dust_seed0.json) | [config](yolo/s26_external_seed0.json) |
| FL | [config](rt/internal_fog_dim_seed0.json) | [config](rt/external_fog_dim_seed0.json) | [config](yolo/s27_external_seed0.json) |
| DL | [config](rt/internal_dust_dim_seed0.json) | [config](rt/external_dust_dim_seed0.json) | [config](yolo/s25_external_seed0.json) |
| RSF | [config](rt/internal_rain_snow_fog_seed0.json) | [config](rt/external_rain_snow_fog_seed0.json) | [config](yolo/s11_external_seed0.json) |
| RSD | [config](rt/internal_rain_snow_dust_seed0.json) | [config](rt/external_rain_snow_dust_seed0.json) | [config](yolo/s09_external_seed0.json) |
| RSL | [config](rt/internal_rain_snow_dim_seed0.json) | [config](rt/external_rain_snow_dim_seed0.json) | [config](yolo/s10_external_seed0.json) |
| RFD | [config](rt/internal_rain_fog_dust_seed0.json) | [config](rt/external_rain_fog_dust_seed0.json) | [config](yolo/s13_external_seed0.json) |
| RFL | [config](rt/internal_rain_fog_dim_seed0.json) | [config](rt/external_rain_fog_dim_seed0.json) | [config](yolo/s14_external_seed0.json) |
| RDL | [config](rt/internal_rain_dust_dim_seed0.json) | [config](rt/external_rain_dust_dim_seed0.json) | [config](yolo/s12_external_seed0.json) |
| SFD | [config](rt/internal_snow_fog_dust_seed0.json) | [config](rt/external_snow_fog_dust_seed0.json) | [config](yolo/s15_external_seed0.json) |
| SFL | [config](rt/internal_snow_fog_dim_seed0.json) | [config](rt/external_snow_fog_dim_seed0.json) | [config](yolo/s16_external_seed0.json) |
| SDL | [config](rt/internal_snow_dust_dim_seed0.json) | [config](rt/external_snow_dust_dim_seed0.json) | [config](yolo/s08_external_seed0.json) |
| FDL | [config](rt/internal_fog_dust_dim_seed0.json) | [config](rt/external_fog_dust_dim_seed0.json) | [config](yolo/s17_external_seed0.json) |
| RSFD | [config](rt/internal_no_dim_low_light_seed0.json) | [config](rt/external_no_dim_low_light_seed0.json) | [config](yolo/s04_external_seed0.json) |
| RSFL | [config](rt/internal_no_dust_seed0.json) | [config](rt/external_no_dust_seed0.json) | [config](yolo/s05_external_seed0.json) |
| RSDL | [config](rt/internal_no_fog_seed0.json) | [config](rt/external_no_fog_seed0.json) | [config](yolo/s03_external_seed0.json) |
| RFDL | [config](rt/internal_no_snow_seed0.json) | [config](rt/external_no_snow_seed0.json) | [config](yolo/s06_external_seed0.json) |
| SFDL | [config](rt/internal_no_rain_seed0.json) | [config](rt/external_no_rain_seed0.json) | [config](yolo/s07_external_seed0.json) |
| RSFDL | [config](rt/internal_full_generation_seed0.json) | [config](rt/external_full_generation_seed0.json) | [config](yolo/s02_external_seed0.json) |

## Frozen references

Train these references with the `rt` and `yolo` families respectively. They use the extra-clean reference split for response measurement.

- [RT-DETRv2-S](rt/reference_seed0.json)
- [YOLO11m](yolo/reference_seed0.json)

## Matched sampling

Use the `ratio` family. Every configuration retains all five generated conditions and changes their total sampling fraction. Files for seeds 0, 1, and 2 are provided.

| Matched subset size | Generated fraction | Seed 0 configuration |
| --- | --- | --- |
| 4 | 4/9 | [config](ratio/full_p4of9_seed0.json) |
| 3 | 3/8 | [config](ratio/full_p0375_seed0.json) |
| 2 | 2/7 | [config](ratio/full_p2of7_seed0.json) |
| 1 | 1/6 | [config](ratio/full_p1of6_seed0.json) |

## Cross-condition transfer

Use the `transfer` family. Each training source has a clean view and a second view from one condition. The control repeats the clean view. Links use seed1; matching seed2 and seed3 files are provided.

| Second view | Seed 1 configuration |
| --- | --- |
| Clean repeat | [config](transfer/seed1_cleanx2.json) |
| Rain | [config](transfer/seed1_rain.json) |
| Snow | [config](transfer/seed1_snow.json) |
| Fog | [config](transfer/seed1_fog.json) |
| Dust | [config](transfer/seed1_dust.json) |
| Low light | [config](transfer/seed1_dim_low_light.json) |
