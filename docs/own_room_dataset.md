# Own-room dataset protocol

The plan requires a small, self-collected set of room photos with hand measurements. Photos
stay local (they may contain private spaces); only the measurement annotations are committed.

## Capture

- Use the same phone/camera if possible, in landscape orientation.
- Take 3–5 photos per room: one standing near a corner, one from the doorway, and 1–3
  additional views that show the floor and all walls.
- Keep the camera roughly at chest height (1.4–1.6 m), level, and avoid extreme wide-angle
  modes unless that is the target use case.
- Use good lighting; avoid people, pets, mirrors, and motion blur.
- Include a photo where doors/windows and the floor-wall boundary are visible.

## Measure (meters)

Record in the annotation file:

- room width and depth (wall to wall),
- door and window positions along each wall, plus their widths,
- each movable item: class, width, depth, center `(x, y)`, and rotation (`0/90/180/270`),
- optional camera height and any scale notes.

## File layout

```
data/raw/rooms/<room_name>/photo_*.jpg          # local, gitignored
data/floorplans_gt/<room_name>/floorplan_gt.json # tracked annotation
```

Start from `data/floorplans_gt/template.json`. Keep the floor-frame convention documented in
`data/floorplans_gt/README.md`.

## Run the evaluation

```bash
python scripts/run_geometry.py --image data/raw/rooms/<room_name>/photo_01.jpg \
    --scale-mode user --room-width <W> --room-depth <D>

python scripts/evaluate_geometry.py \
    --predictions results/geometry \
    --ground-truth data/floorplans_gt
```

For a multi-photo room, use the best single view for the current single-image pipeline and
record the same ground truth under that prediction's directory name.

## Reporting

The evaluator reports room width/depth/area errors, furniture center/dimension errors, footprint
IoU, and matched/false-positive/false-negative counts. Include these numbers in the midterm
report and the E2 scale-mode comparison.
