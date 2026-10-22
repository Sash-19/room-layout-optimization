# Ground-truth floor plans

One directory per measured room, named to match the prediction directory:

```
data/floorplans_gt/<room_name>/floorplan_gt.json
```

`floorplan_gt.json` uses the same schema as `results/geometry/<room_name>/floorplan.json`
(see `template.json`). Coordinates are meters in a floor frame:

- `x` points right along the `min_y` wall, `y` points into the room.
- The room polygon is axis-aligned for the current rectangle-only representation.
- `theta_deg` is the furniture rotation around its center.
- Wall indices are `0=min_y`, `1=max_x`, `2=max_y`, `3=min_x`.
- Door/window offsets are measured from the wall's minimum coordinate.

Only the `room` object is required for geometry evaluation; scale metadata is optional.

Run:

```bash
python scripts/evaluate_geometry.py \
    --predictions results/geometry \
    --ground-truth data/floorplans_gt
```

This writes `results/geometry_eval/metrics.json` and `metrics.md`.
