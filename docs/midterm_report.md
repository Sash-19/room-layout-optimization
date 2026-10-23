# Midterm Report: 2D Room Layout Representation

**Status:** draft. The pipeline, scorer, and optimizer are implemented and smoke-tested. E1 now
has a measured comparison on a small Open Images subset; E2 and geometry-accuracy results still
need the measured room data described in Section 5.

## 1. Scope

This checkpoint covers the first half of the project: turning a single room photograph into a
structured 2D floor plan, checking that plan against hard constraints, scoring it, and searching
for better furniture arrangements.

Implemented at this checkpoint:

- Unified perception: furniture/door/window detection, monocular depth, floor/wall masks.
- Geometry: floor-plane fitting, metric scale calibration, rectangular room estimation,
  furniture footprint lifting, opening placement.
- Representation: JSON floor plan with rooms, furniture, openings, and metric units.
- Scoring: 5 cm occupancy grid, hard constraints, soft terms, and a weighted score.
- Optimization: random feasible search, greedy sequential placement, and simulated annealing
  with restarts and top-K layouts.
- Evaluation tooling: unified-class detector mAP, geometry accuracy, and E1/E2/E3 experiment
  harnesses.

## 2. Architecture

![System pipeline](assets/architecture.png)

Data flow:

1. `PerceptionPipeline` produces detections, a depth map, and floor/wall masks.
2. `GeometryPipeline` fits the floor plane, estimates scale, estimates a rectangular room, and
   projects object footprints onto the floor plane.
3. `FloorPlan` is serialized to JSON.
4. `LayoutScorer` rasterizes furniture, runs hard constraints, computes soft terms, and returns
   a score plus feasibility violations.
5. The optimizer perturbs furniture poses and uses the same scorer to compare candidates.

## 3. Representation and scoring

- Coordinate frame: camera x right, y down, z forward; floor frame `u` right and `v` forward.
- Units: meters after scale calibration.
- Openings: `wall_index` 0=min_y, 1=max_x, 2=max_y, 3=min_x.
- Grid: 5 cm cells. Furniture is rasterized by its footprint; clearance is handled by scoring
  terms rather than by inflating the footprint.
- Hard constraints: inside room, pairwise furniture overlap, door clear-zone blocking, and full
  window blocking.
- Soft terms: overlap, walking area, path quality, wall proximity, door/window clearance,
  utilization, clearance, and pair preferences.
- Score: weighted sum in `[0, 1]`; feasibility is reported separately. The default weights sum
  to 1.0, with walking area (0.22), door clearance (0.18), and front clearance (0.16) dominant.

Walkability is a soft term, not a hard constraint. A layout can therefore be feasible while
still receiving a low walking score.

## 4. Experimental setup

### 4.1 Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cpu (CUDA unavailable in the smoke environment) |
| Ultralytics | 8.4.184 |
| Transformers | 5.20.2 |
| Shapely | 2.2.0 |
| NumPy | 2.5.2 |

All reported runs were CPU-only. Fine-tuning is intended for a GPU; the implementation does not
require one for inference of the small models.

### 4.2 Perception

- Detector: YOLOv8s, 640 px input, confidence 0.25, NMS IoU 0.7, maximum 300 detections.
- Unified taxonomy: bed, desk, chair, couch, table, cabinet, door, window.
- Depth: Depth Anything V2 Small for relative depth; Depth Anything V2 Metric Indoor Small for
  the metric-depth smoke check.
- Segmentation: SegFormer-B0 fine-tuned on ADE20K; floor label 3, wall label 0.

### 4.3 Training and evaluation defaults

- Split: 70% train, 15% validation, 15% test; seed 0.
- Training: 100 epochs, batch 16, 640 px, initial learning rate 0.01, cosine schedule,
  patience 20.
- Detection evaluation: confidence 0.001 for AP, report confidence 0.25, IoU thresholds
  0.50:0.05:0.95.

These are configuration defaults. The current E1 run used a smaller setup: 20 epochs, batch 16,
640 px, on the 400-image Open Images subset, because the full dataset is much larger than the
available CPU-only environment. The reported E1 numbers are therefore a small-subset result, not
a full-benchmark result.

### 4.4 Geometry, scoring, and optimization

- Floor plane: RANSAC with 300 iterations, 0.03 inlier threshold, stride 4.
- Scale modes: metric depth, known reference, user room dimensions, or unscaled.
- Scoring: 5 cm grid, 0.6 m walkway width, 5 cm ideal wall gap, 0.5 m wall tolerance,
  0.9 m door clear depth, 0.3 m window clear depth.
- Random search: 300 feasible layouts.
- Greedy: 120 candidate poses per item, wall-affinity and large items placed first.
- Simulated annealing: 600 iterations, 2 restarts, temperature 0.05 to 0.001, 0.4 m translation
  step, top-3 layouts, warm start from greedy.

## 5. Dataset status

The detection dataset is now populated; the measured room data is still missing.

| Dataset | Status |
|---|---|
| Detection training set (`data/processed`) | 400 Open Images V5 validation images; 280/60/60 train/val/test; 1,999 instances |
| Own-room photographs and measurements | Not collected; only the capture protocol and GT template exist |
| Smoke photographs | Two COCO room images (`000000000139`, `000000000632`) with cached perception outputs |

The detection subset is downloaded with `scripts/download_openimages_subset.py`, which selects
images containing the unified classes and writes YOLO labels. It is merged into the unified
taxonomy with the 70/15/15 split. Class counts are: bed 60, desk 170, chair 493, couch 51,
table 417, cabinet 195, door 133, window 480 across all splits. The subset is intentionally small
for CPU-only iteration; the full planned sources are COCO, Open Images, and Roboflow room data.
The own-room protocol is documented in `docs/own_room_dataset.md`; ground-truth JSON uses
`data/floorplans_gt/<room>/floorplan_gt.json`.

Own-room capture plan: 3-5 landscape photos per room at roughly chest height, including a
doorway view and views that show the floor-wall boundary. Hand measurements cover room width and
depth, door/window offsets and widths, and each movable item's class, dimensions, center, and
0/90/180/270 rotation. Photos stay local; only the measurement annotations are committed.

## 6. Experiment design

| ID | Question | Inputs | Metrics |
|---|---|---|---|
| E1 | Does fine-tuning beat zero-shot detection? | Open Images room subset | mAP@0.5, mAP@0.5:0.95, P, R, F1 |
| E2 | Which scale source is accurate enough? | Single photos + depth + measured room | room width/depth error, furniture position error, feasibility |
| Geometry | How accurate is the lifted plan? | Own-room photos + hand measurements | room error, furniture center/dim error, footprint IoU |
| E3 | Does optimization improve layouts? | Detected and synthetic floor plans | score delta, feasibility rate, evaluations, runtime |

Detector metrics remap both zero-shot and fine-tuned predictions to the unified taxonomy before
matching. Geometry evaluation matches furniture by class and footprint overlap, then reports
room errors, center/dimension errors, and footprint IoU. E2 adds hand-measured room and furniture
errors; without those measurements the error columns are intentionally `n/a`.

Scale modes in E2:

- `metric`: metric depth model; no user dimensions.
- `reference`: known-size object or opening; no user dimensions.
- `user`: user supplies room width/depth.

## 7. Results status

### 7.1 Reportable results

| Experiment | Status | Blocker |
|---|---|---|
| E1 detector comparison | Measured on 400-image subset | Larger/GPU training run for a stronger benchmark |
| E2 scale comparison | Smoke only | Measured room/furniture ground truth; working reference mode |
| Geometry accuracy | Pending | Own-room photos and hand measurements |
| E3 optimization comparison | Smoke only | Larger set of detected and synthetic floor plans |

The E1 table below is a real comparison on the prepared subset. The numbers in Section 7.3 are
pipeline checks, not benchmark results.

### 7.2 E1 detector comparison

Zero-shot YOLOv8s vs. YOLOv8s fine-tuned for 20 epochs on the 400-image subset, evaluated on the
60-image test split:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| zero-shot | 0.2260 | 0.1957 | 0.2724 | 0.2500 | 0.2607 |
| fine-tuned | 0.3025 | 0.1861 | 0.4257 | 0.3630 | 0.3919 |

Fine-tuning raises mAP@0.5 by 0.0765, precision by 0.1533, recall by 0.1130, and F1 by 0.1312.
mAP@0.5:0.95 is slightly lower (-0.0096), so the gain is concentrated at the looser IoU
threshold and the subset is too small to claim a broad detector improvement.

### 7.3 Smoke checks

Relative-depth perception on `000000000632` (bed + chair):

| Mode | Scale | Room (m) | Camera height (m) | Feasible | Score |
|---|---:|---|---:|---|---:|
| metric | 1.000 | 1.50 x 2.30 | 1.00 | no | 0.886 |
| reference | 1.000 | 1.50 x 2.30 | 1.00 | no | 0.886 |
| user (3.5 x 4.0) | 2.037 | 3.50 x 4.00 | 2.04 | no | 0.909 |

Metric-depth perception on the same image:

| Mode | Scale | Room (m) | Camera height (m) | Feasible | Score |
|---|---:|---|---:|---|---:|
| metric | 1.000 | 1.50 x 1.76 | 1.60 | no | 0.867 |
| reference | 1.000 | 1.50 x 1.76 | 1.60 | no | 0.867 |
| user (3.5 x 4.0) | 2.302 | 3.50 x 4.00 | 3.67 | no | 0.875 |

Relative-depth perception on `000000000139` (table + 3 chairs): score `0.8885`, infeasible due
to chair/table overlaps. `000000000632`: score `0.9085`, infeasible because the bed and chair
fall outside the estimated room.

Optimizer smoke check on the `000000000632` plan:

| Algorithm | Initial | Best | Improvement | Feasible |
|---|---:|---:|---:|---|
| random search | 0.9085 | 0.9091 | +0.0005 | yes |
| greedy | 0.9085 | 0.9129 | +0.0044 | yes |
| simulated annealing | 0.9085 | 0.9144 | +0.0059 | yes |

E3 aggregate over the two smoke floor plans:

| Algorithm | Mean initial | Mean best | Mean delta | Improved | Feasible init -> best | Mean runtime (s) |
|---|---:|---:|---:|---:|---|---:|
| random | 0.8985 | 0.9114 | +0.0129 | 2/2 | 0 -> 2 | 1.46 |
| greedy | 0.8985 | 0.9197 | +0.0212 | 2/2 | 0 -> 2 | 1.31 |
| simulated annealing | 0.8985 | 0.9204 | +0.0219 | 2/2 | 0 -> 2 | 9.09 |

![Detected furniture](assets/perception_632_detections.jpg)

*Perception smoke check: bed and chair detected in `000000000632`.*

![Relative depth](assets/depth_632.png)

*Relative-depth output. Larger values are closer; geometry calibrates the inverse depth against
the floor plane.*

![Floor/wall structure](assets/structures_632.jpg)

*SegFormer floor/wall overlay used by the geometry stage.*

![Detected floor plan](assets/geometry_632.png)

*Detected floor plan. The bed lies partly outside the estimated room, which is why the initial
layout is infeasible.*

![Before and after optimization](assets/optimizer_before_after.png)

*Optimizer smoke check: the detected layout is moved inside the room and becomes feasible. The
score gain is small because the initial furniture arrangement was already reasonable.*

![Failure case](assets/geometry_139.png)

*Failure case: the visible floor is partial, the room estimate is too small, and furniture falls
outside it.*

## 8. Failure cases

- A single photo often shows only part of the floor, so the rectangular room can be too small.
  Both smoke images produced outside-room violations.
- Relative depth is inverse depth; geometry calibrates it against the floor plane, but absolute
  scale still depends on a reference or user dimensions.
- With no door/window detection, the `reference` mode cannot calibrate scale.
- Sparse furniture layouts can make the walking-connectivity term fail even when the visible
  area looks reasonable.
- Detection class coverage and small-object recall are the main perception risks.
- The optimizer can exploit scoring terms, for example by hugging walls or leaving large empty
  regions, and its score gains on already-reasonable layouts are small.
- Simulated annealing is much slower than greedy on the smoke plans (about 6-7x) for a modest
  score improvement; the current schedule is not yet tuned.
- E2 and geometry scores still lack measured room/furniture ground truth.

## 9. Open issues

- E1 uses a 400-image subset and 20 CPU epochs; a larger/GPU run is needed before strong claims.
- Own-room photos and hand measurements are needed for geometry accuracy.
- Metric-depth inference works, but the room estimate still needs measured ground truth.
- Reference-scale calibration needs detected or annotated openings.
- GA comparison, larger E3 runs, and end-to-end visualization are second-half work.
- The report needs a final pass with measured E2/geometry tables, figure numbering, and submission formatting.

## 10. Next steps

1. Scale up or GPU-train E1 and rerun the comparison on a held-out split.
2. Capture own-room photos and measurements; run `scripts/evaluate_geometry.py`.
3. Measure rooms for E2 and fix reference-scale calibration.
4. Run E3 on a larger set, including synthetic bad layouts and before/after figures.
5. Add the GA comparison and end-to-end image-to-optimized-plan CLI.
6. Finalize the report with measured E2/geometry/E3 tables and citations.

## 11. Conclusion

The midterm checkpoint delivers a working 2D representation and a reproducible evaluation
scaffold: perception, geometry lifting, constraint checking, scoring, and three optimizers all
run end-to-end on smoke images. E1 now has a measured subset result showing a higher mAP@0.5
after fine-tuning, but E2 and geometry accuracy still need measured room data, so the report
does not yet claim scale or geometry accuracy.

## 12. References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified,
  Real-Time Object Detection.* CVPR.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer:
  Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS.
- Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). *Depth Anything V2.*
  NeurIPS.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing.*
  Science, 220(4598), 671-680.
- Merrell, P., Schkufza, E., & Koltun, V. (2011). *Computer-Generated Residential Building
  Layouts.* ACM Transactions on Graphics.
- Paschalidou, D., Kar, A., Shugrina, M., Kreis, K., & Fidler, S. (2021). *ATISS: Autoregressive
  Transformers for Indoor Scene Synthesis.* NeurIPS.
- Ultralytics. *YOLOv8 / YOLO11 documentation.* https://docs.ultralytics.com

Citations should be rechecked against the publisher or arXiv records before final submission.

## Appendix: commands

```bash
# Data setup
python scripts/download_openimages_subset.py --out data/raw/openimages
python scripts/prepare_dataset.py --config configs/detection.yaml --copy-mode link

# E1
python scripts/run_e1_detector.py --config configs/detection.yaml \
    --data data/processed --split test \
    --weights yolov8s.pt runs/detect/runs/detect/openimages_20e/weights/best.pt \
    --labels zero_shot fine_tuned

# E2
python scripts/run_e2_scale.py --perception-dir results/perception/room1 \
    --room-width 3.5 --room-depth 4.0

# Geometry accuracy
python scripts/evaluate_geometry.py \
    --predictions results/geometry \
    --ground-truth data/floorplans_gt

# Optimizer comparison
python scripts/optimize_floorplan.py \
    --floorplan results/geometry/room1/floorplan.json --algorithm all

# E3
python scripts/run_e3_optimization.py \
    --floorplans results/geometry \
    --algorithms random greedy annealing
```
