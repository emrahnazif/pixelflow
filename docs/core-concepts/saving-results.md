# Save results to JSON and load them back

<!--
summary: Write detections and classifications to JSON or plain dictionaries, read them back, and keep the file small by storing outlines instead of masks.
label: Saving results
keywords: pixelflow, save detections, json, serialization, segmentation masks
-->

Results often need to outlive the script that made them: written to a file, stored in a
database, or sent to a web page. `to_json()` turns detections into a JSON string, and
`Detections.from_json()` reads it back into a `Detections` that filters, tracks and draws like
one that came straight from a model.

```python
from pathlib import Path

import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-seg")
result = predictor("street.jpg")
detections = pf.from_mayaku(result, labels=predictor.class_names)
detections = detections.filter_by_confidence(0.5)

Path("street.json").write_text(detections.to_json())

restored = pf.Detections.from_json(Path("street.json").read_text())
print(len(restored), restored[0].class_name, restored[0].masks[0].shape)
```

```text
11 Person (1280, 1920)
```

Everything comes back: boxes, masks, outlines, keypoints, classes, text, tracker ids, zones,
line crossings and times.

## JSON or dictionaries

| Write | Read | Gives |
|-------|------|-------|
| `detections.to_json()` | `pf.Detections.from_json(text)` | a JSON string, indented for reading |
| `detections.to_dict()` | `pf.Detections.from_dict(items)` | a list of dictionaries, one per detection |

Use `to_dict()` when the result goes somewhere that takes Python data: a web framework that
writes the JSON response itself, a database driver, or a pandas `DataFrame`. Its values are plain
Python types (no NumPy numbers), so `json.dumps(detections.to_dict())` always works, and gives
compact JSON without the indentation of `to_json()`.

Each dictionary has every field of a [detection](/docs/core-concepts/detections.md), with
`None` for the empty ones, so every record has the same keys:

```json
{
    "bbox": [4.56, 588.91, 333.44, 1273.22],
    "class_id": 0,
    "class_name": "Person",
    "confidence": 0.835,
    "masks": [{"format": "png", "data": "iVBORw0KGgo..."}],
    "segments": null,
    "keypoints": null,
    "tracker_id": null,
    ...
}
```

## Masks or outlines

A mask is stored as a PNG image inside the JSON, so it comes back pixel for pixel. When you only
need the shape, store the outline instead: `masks_to_segments(keep_masks=False)` traces each
mask into a polygon and drops the mask.

```python
import json

detections.masks_to_segments(keep_masks=False)
compact = json.dumps(detections.to_dict())
```

Measured on the 11 people above (masks of 1920×1280 pixels), as compact JSON:

| Stored | Size |
|--------|------|
| masks | 30 KB |
| outlines | 13 KB |

An outline is a list of points that a web page can draw as it is, without decoding an image.
It follows the mask's edge to within `tolerance` pixels (default 1.5), so it is close to the
mask but not identical. Keep the masks when you need the exact pixels.

To trace a single mask yourself, `pf.mask_to_polygon(mask, tolerance=1.5)` returns its outline as
`[[x, y], ...]`.

## Classifications

[Classifications](/docs/core-concepts/classifications.md) have the same four methods:
`result.to_json()`, `result.to_dict()`, `pf.Classifications.from_json(text)` and
`pf.Classifications.from_dict(items)`.
