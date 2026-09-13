# Map evaluation: best.pt

Model `pointnet`, threshold 0.71, 2599 windows of lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap.

Operational band [-0.1, 0.6] m about the measured floor, 8.0 m range, latest value per 0.05 m voxel, reduced to 0.1 m ground cells.

| measure | control | cleaned |
|---|---|---|
| ground_cells | 2561 | 2436 |
| false_cells_3d | 2211 | 2074 |
| false_cells_footprint | 2195 | 2068 |
| false_cell_seconds | 6399138.0000 | 6267844.7000 |
| macro_coverage | 0.6499 | 0.6834 |
| worst_rock_coverage | 0.2941 | 0.2941 |
| rocks_never_covered | 0 | 0 |
| rocks_with_lost_cells | 12 | 12 |
| median_first_covered_s | 158.3702 | 158.3702 |
| slowest_first_covered_s | 497.7793 | 497.7793 |
| retracted_voxels | 0 | 13023 |

## Per rock

| rock | control coverage | cleaned coverage | control first seen (s) | cleaned first seen (s) |
|---|---|---|---|---|
| 1 | 0.731 | 0.731 | 158 | 158 |
| 3 | 0.776 | 0.816 | 159 | 159 |
| 4 | 0.727 | 0.705 | 161 | 161 |
| 5 | 0.621 | 0.621 | 158 | 158 |
| 6 | 0.787 | 0.787 | 169 | 169 |
| 7 | 0.773 | 0.864 | 168 | 168 |
| 8 | 0.825 | 0.825 | 498 | 498 |
| 9 | 0.724 | 0.724 | 158 | 158 |
| 10 | 0.636 | 0.636 | 158 | 158 |
| 11 | 0.500 | 0.650 | 158 | 158 |
| 12 | 0.294 | 0.294 | 0 | 0 |
| 13 | 0.405 | 0.548 | 156 | 156 |

Coverage denominators are ground cells a real return actually landed on inside each rock, unfiltered and shared by both maps.

`threshold-curve.csv` is the same map read at every threshold, so two checkpoints can be compared at equal wrongly-claimed area rather than at whichever threshold each happens to carry. A threshold chosen by reading it is an oracle diagnostic, not an operating point.

`map.png` is the same thing looking down: aggregates cannot show whether the wrongly-claimed ground is scattered or piled in two places.
