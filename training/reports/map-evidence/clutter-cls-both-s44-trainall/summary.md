# Map evaluation: best.pt

Model `pointnet`, threshold 0.72, 2599 windows of lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap.

Operational band [-0.1, 0.6] m about the measured floor, 8.0 m range, latest value per 0.05 m voxel, reduced to 0.1 m ground cells.

| measure | control | cleaned |
|---|---|---|
| ground_cells | 2485 | 2406 |
| false_cells_3d | 2122 | 2041 |
| false_cells_footprint | 2121 | 2040 |
| false_cell_seconds | 5464396.2000 | 5361358.0000 |
| macro_coverage | 0.6938 | 0.6910 |
| worst_rock_coverage | 0.3636 | 0.3636 |
| rocks_never_covered | 0 | 0 |
| rocks_with_lost_cells | 12 | 12 |
| median_first_covered_s | 158.3702 | 158.3702 |
| slowest_first_covered_s | 499.4274 | 499.4274 |
| retracted_voxels | 0 | 5703 |

## Per rock

| rock | control coverage | cleaned coverage | control first seen (s) | cleaned first seen (s) |
|---|---|---|---|---|
| 1 | 0.808 | 0.808 | 158 | 158 |
| 3 | 0.816 | 0.816 | 159 | 159 |
| 4 | 0.727 | 0.727 | 160 | 160 |
| 5 | 0.621 | 0.621 | 158 | 158 |
| 6 | 0.809 | 0.809 | 169 | 169 |
| 7 | 0.818 | 0.818 | 168 | 168 |
| 8 | 0.825 | 0.825 | 499 | 499 |
| 9 | 0.828 | 0.793 | 158 | 158 |
| 10 | 0.364 | 0.364 | 158 | 158 |
| 11 | 0.550 | 0.550 | 158 | 158 |
| 12 | 0.471 | 0.471 | 0 | 0 |
| 13 | 0.690 | 0.690 | 156 | 156 |

Coverage denominators are ground cells a real return actually landed on inside each rock, unfiltered and shared by both maps.

`threshold-curve.csv` is the same map read at every threshold, so two checkpoints can be compared at equal wrongly-claimed area rather than at whichever threshold each happens to carry. A threshold chosen by reading it is an oracle diagnostic, not an operating point.

`map.png` is the same thing looking down: aggregates cannot show whether the wrongly-claimed ground is scattered or piled in two places.
