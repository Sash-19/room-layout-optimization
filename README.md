# Room Layout Optimization Using Computer Vision

Takes a single room photograph, detects the furniture and architectural elements, lifts them
into a metric 2D floor plan, and searches for better furniture arrangements with an explicit
constraint-based scoring function.

The full plan lives in [`../project plan.md`](../project%20plan.md).

## Status

| Stage | Module | State |
|-------|--------|-------|
| 1. Perception (detection, segmentation, depth) | `src/perception/` | implemented |
| 2. Geometry lifting (camera, floor plane, scale) | `src/geometry/` | implemented |
| 3. Floor plan representation | `src/floorplan/` | implemented |
| 4. Scoring + layout optimization | `src/scoring/`, `src/optimization/` | scoring implemented; optimizer not started |
| 5. Visualization / end-to-end pipeline | `src/viz/`, `src/pipeline.py` | not started |

This repository currently implements **Stages 1–3 plus the scoring half of Stage 4**: perception,
geometry lifting, the floor-plan representation/priors, and a constraint-based layout scorer.
Everything downstream has a stable data contract (`src/schemas.py`,
`src/floorplan/representation.py`) so models and components can be swapped without touching
later stages.

## What the perception stage produces

Given an RGB image, the pipeline outputs:

- **Detections** in a unified taxonomy (`bed, desk, chair, couch, table, cabinet, door, window`)
  as `Detection(cls, conf, bbox_xyxy, mask?)`.
- A **monocular depth map** (`DepthMap`), metric or normalized-relative.
- **Floor / wall masks** (`StructureMasks`) for the geometry stage.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate roomlayout
```

A GPU is recommended for fine-tuning but not required for inference of the small models.
The training/eval scripts are pure Python and unit-tested without torch installed.

## Unified class taxonomy

```
movable:  bed, desk, chair, couch, table, cabinet (incl. wardrobe, bookcase, dresser, nightstand)
fixed:    door, window
```

Source names are mapped to these classes with alias rules in `src/perception/taxonomy.py`
(e.g. `sofa -> couch`, `windowpane -> window`, `dining table -> table`). Source-specific
overrides go in a `class_map` in the config.

## 1. Prepare the dataset

Merge COCO / Open Images / Roboflow sources into one YOLO-format dataset with the unified
taxonomy and a 70/15/15 split. Fill in `data.sources` in `configs/detection.yaml`:

```yaml
data:
  output: data/processed
  split_ratios: {train: 0.70, val: 0.15, test: 0.15}
  sources:
    - name: coco_rooms
      type: coco
      images: data/raw/coco/train2017
      annotations: data/raw/coco/annotations/instances_train2017.json
      class_map:
        bed: bed
        chair: chair
        couch: couch
        dining table: table
    - name: openimages_rooms
      type: yolo
      root: data/raw/openimages/yolo
      names: [Sofa, Cupboard, Wardrobe, Door, Window]
      class_map:
        Sofa: couch
        Cupboard: cabinet
        Wardrobe: cabinet
        Door: door
        Window: window
      split: train
```

Then:

```bash
python scripts/prepare_dataset.py --config configs/detection.yaml
```

Output (`data/processed/`): `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`,
`manifest.json`. Unmapped classes are dropped; add a `null` value in `class_map` to
explicitly drop a class. Use `--keep-splits` to preserve a source's own split, `--copy-mode link`
to hardlink instead of copy, and `--verify-images` to skip corrupt images.

## 2. Train the detector

```bash
python scripts/train_detector.py --config configs/detection.yaml
```

Starts from `model.weights` (default `yolov8s.pt`), fine-tunes with the settings under `train:`,
and writes weights to `runs/detect/train/weights/best.pt`. All `train:` keys are passed straight
to Ultralytics, so any YOLO argument can be added. Override on the CLI with `--weights`,
`--epochs`, `--batch`, `--imgsz`, `--device`, `--project`, `--name`.

## 3. Evaluate the detector

Measurements use a custom evaluator so **zero-shot COCO weights** and **fine-tuned unified
weights** can be compared apples-to-apples: predictions are remapped to the unified taxonomy
and matched against the unified ground truth.

```bash
# fine-tuned model
python scripts/evaluate_detector.py --config configs/detection.yaml \
    --weights runs/detect/train/weights/best.pt --split test

# zero-shot baseline (E1)
python scripts/evaluate_detector.py --config configs/detection.yaml \
    --weights yolov8s.pt --split test --name zeroshot
```

Reports mAP@0.5, mAP@0.5:0.95, precision, recall, F1 and per-class AP@0.5, in the console,
as JSON and as a Markdown table under `results/detection_eval/`.

Run both baselines in one table (E1):

```bash
python scripts/run_e1_detector.py --config configs/detection.yaml \
    --data data/processed --split test \
    --weights yolov8s.pt runs/detect/train/weights/best.pt \
    --labels zero_shot fine_tuned
```

Writes per-model metrics plus `e1_detector.json` / `e1_detector.md` under
`results/experiments/e1_detector/`.

## 4. Run the perception demo

```bash
python scripts/run_perception.py --config configs/perception.yaml \
    --image data/samples/bedroom1.jpg

# detection only (no depth/segmentation models)
python scripts/run_perception.py --image data/samples/bedroom1.jpg --no-depth --no-structure

# use a trained detector
python scripts/run_perception.py --image data/samples/bedroom1.jpg \
    --weights runs/detect/train/weights/best.pt
```

Writes per image to `results/perception/<name>/`:

```
input.jpg            detections.jpg        detections.json
depth.png            depth.npy             floor_mask.png
wall_mask.png        structures.jpg        result.json
```

`configs/perception.yaml` controls the detector, depth model (`depth-anything/Depth-Anything-V2-Small-hf`
by default, relative depth) and floor/wall segmenter (`segformer-b0` ADE20K). Set `metric: true`
and a metric depth checkpoint to get depth in meters.

## 5. Lift perception into a floor plan

```bash
# run perception and geometry in one call
python scripts/run_geometry.py --config configs/geometry.yaml \
    --image data/samples/bedroom1.jpg

# reuse cached perception output
python scripts/run_geometry.py --perception-dir results/perception/bedroom1 \
    --scale-mode user --room-width 3.5 --room-depth 4.0

# absolute scale from a known reference object instead
python scripts/run_geometry.py --perception-dir results/perception/bedroom1 \
    --scale-mode reference --reference-class door --reference-width 0.8
```

The geometry stage:

- builds camera intrinsics from explicit config values, image EXIF, or a fallback FOV;
- fits the floor plane with RANSAC on back-projected floor-mask depth;
- calibrates Depth Anything's **relative inverse depth** against the floor so the plane is
  metric-consistent up to one scale factor (metric depth skips this step);
- estimates scale from `metric`, `user` room dimensions, a `reference` object width, or leaves
  the plan `unscaled`;
- projects each detection's bottom edge onto the floor, snaps orientation, and applies
  furniture priors (`src/floorplan/priors.py`);
- estimates the rectangular room boundary and assigns openings to walls.

Writes to `results/geometry/<name>/`:

```
floorplan.json       floorplan.png
```

`floorplan.json` contains the room polygon, furniture footprints (meters, floor-frame
coordinates), openings, scale mode/scale factor, camera height, and fit diagnostics/warnings.
Relative depth is scale-free by nature, so pass `--room-width`/`--room-depth` (or a reference
width) for metric output.

## 6. Score a floor plan

```bash
python scripts/score_floorplan.py \
    --floorplan results/geometry/bedroom1/floorplan.json \
    --config configs/scoring.yaml
```

Writes `score.json` next to the floor plan with:

- total weighted score in `[0, 1]` and per-term breakdown;
- hard-constraint violations (outside room, footprint overlap, blocked door, blocked window);
- layout metrics: room/free/walking area, usable-floor fraction, overlap pairs/area, door
  clearance, largest free component.

Scoring uses a 5 cm occupancy grid (`src/scoring/grid.py`) with A* path lengths and distance
transforms. Soft terms are `overlap`, `walk`, `path`, `wall`, `door`, `window`, `util`,
`clear`, and optional `pair`; weights and thresholds live in `configs/scoring.yaml`.

## 7. Measure geometry accuracy

Hand-annotate measured rooms under `data/floorplans_gt/<room>/floorplan_gt.json` (see
`data/floorplans_gt/README.md` and `template.json`), then:

```bash
python scripts/evaluate_geometry.py \
    --predictions results/geometry \
    --ground-truth data/floorplans_gt
```

Writes `results/geometry_eval/metrics.json` and `metrics.md` with room width/depth/area errors,
furniture center/dimension errors, footprint IoU, and matched/false-positive/false-negative
counts. The capture and annotation protocol is in `docs/own_room_dataset.md`.

## 8. Run the scale-calibration experiment (E2)

```bash
# relative-depth perception, compare all modes
python scripts/run_e2_scale.py \
    --perception-dir results/perception/bedroom1 \
    --room-width 3.5 --room-depth 4.0

# metric-depth perception
python scripts/run_perception.py --config configs/perception_metric.yaml \
    --image data/samples/bedroom1.jpg
python scripts/run_e2_scale.py \
    --perception-dir results/perception_metric/bedroom1 \
    --modes metric,reference,user --room-width 3.5 --room-depth 4.0
```

Each mode writes a floor plan, score JSON, and a combined `e2_scale.json` / `e2_scale.md`
table. Pass `--ground-truth data/floorplans_gt/<room>/floorplan_gt.json` to add room/furniture
accuracy columns.

## Tests

```bash
python -m pytest -q
```

Covers schemas, taxonomy/aliasing, YOLO result post-processing, depth/structure post-processing,
dataset merging/splitting, detection evaluation (matching + AP), perception/geometry pipelines,
camera/floor-plane/scale estimation, lifting, the floor-plan representation/priors,
occupancy-grid constraints/scoring terms, geometry accuracy evaluation, and visualization. The
suite runs without torch/ultralytics installed; geometry/scoring tests need `shapely`, `scipy`,
and `matplotlib` from `requirements.txt`.

## Module map

```
src/
├── schemas.py                 Detection, DepthMap, StructureMasks
├── utils.py                   YAML/JSON/image/device helpers
├── dataset.py                 COCO/YOLO -> unified YOLO dataset
├── perception/
│   ├── taxonomy.py            unified classes + aliases
│   ├── detector.py            Ultralytics YOLO wrapper (train/infer)
│   ├── evaluation.py          unified-class mAP / P / R / F1
│   ├── depth.py               monocular depth (Depth Anything V2)
│   ├── structure.py           floor/wall segmentation (SegFormer ADE20K)
│   ├── cache.py               reload cached perception outputs
│   ├── pipeline.py            detection + depth + structure
│   └── visualize.py           boxes, masks, depth colormaps
├── geometry/
│   ├── camera.py              intrinsics from config/EXIF/FOV, projection
│   ├── floor_plane.py         RANSAC plane fit, inverse-depth calibration
│   ├── room_boundary.py       floor frame + rectangular room estimate
│   ├── lift.py                bottom-edge projection, furniture/openings
│   ├── scale.py               metric/user/reference/unscaled modes
│   ├── pipeline.py            perception -> FloorPlan
│   ├── evaluation.py          geometry accuracy vs hand measurements
│   └── visualize.py           2D floor-plan rendering
├── floorplan/
│   ├── representation.py      Room, Furniture, Opening, FloorPlan
│   └── priors.py              furniture/opening dimensions and clamps
└── scoring/
    ├── grid.py                occupancy grid, distance transform, A*
    ├── constraints.py         hard constraints / violations
    ├── terms.py               soft scoring terms
    ├── zones.py               door/window/clearance polygons
    └── score.py               weighted total + metrics + JSON
```

## Roadmap

Next stage is **layout generation/optimization** (`src/optimization/`): random feasible and
wall-snapped sampling, greedy placement, simulated annealing with restarts/top-K diversity, and
a DEAP genetic-algorithm comparison. The scorer already exposes per-term scores, hard-constraint
violations, and layout metrics so candidate layouts can be compared apples-to-apples.
