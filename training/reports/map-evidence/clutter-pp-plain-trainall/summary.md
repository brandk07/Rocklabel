# Map evaluation: best.pt

Model `pointnet2`, threshold 0.74, 2599 windows of lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap.

Operational band [-0.1, 0.6] m about the measured floor, 8.0 m range, latest value per 0.05 m voxel, reduced to 0.1 m ground cells.

| measure | control | cleaned |
|---|---|---|
| ground_cells | 2043 | 1900 |
| false_cells_3d | 1690 | 1548 |
| false_cells_footprint | 1686 | 1546 |
| false_cell_seconds | 5192042.9000 | 5019169.7000 |
| macro_coverage | 0.6754 | 0.6800 |
| worst_rock_coverage | 0.2941 | 0.3529 |
| rocks_never_covered | 0 | 0 |
| rocks_with_lost_cells | 12 | 12 |
| median_first_covered_s | 158.3702 | 158.3702 |
| slowest_first_covered_s | 497.7793 | 497.7793 |
| retracted_voxels | 0 | 3929 |

## Per rock

| rock | control coverage | cleaned coverage | control first seen (s) | cleaned first seen (s) |
|---|---|---|---|---|
| 1 | 0.769 | 0.769 | 158 | 158 |
| 3 | 0.714 | 0.735 | 159 | 159 |
| 4 | 0.636 | 0.636 | 161 | 161 |
| 5 | 0.724 | 0.724 | 158 | 158 |
| 6 | 0.766 | 0.766 | 169 | 169 |
| 7 | 0.818 | 0.818 | 168 | 168 |
| 8 | 0.800 | 0.800 | 498 | 498 |
| 9 | 0.690 | 0.690 | 158 | 158 |
| 10 | 0.455 | 0.455 | 158 | 158 |
| 11 | 0.700 | 0.700 | 158 | 158 |
| 12 | 0.294 | 0.353 | 0 | 0 |
| 13 | 0.738 | 0.714 | 156 | 156 |

Coverage denominators are ground cells a real return actually landed on inside each rock, unfiltered and shared by both maps.

`threshold-curve.csv` is the same map read at every threshold, so two checkpoints can be compared at equal wrongly-claimed area rather than at whichever threshold each happens to carry. A threshold chosen by reading it is an oracle diagnostic, not an operating point.

`map.png` is the same thing looking down: aggregates cannot show whether the wrongly-claimed ground is scattered or piled in two places.
