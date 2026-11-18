# Detections: one format for every model's results

<!--
summary: What a Detections object holds, how to read boxes, classes, masks and keypoints from it, and how to build one yourself.
label: Detections
keywords: pixelflow, detections, bounding box, segmentation mask, keypoints
-->

A detection model tells you what it found and where. Each framework returns that in its own
objects, with its own field names and box formats. PixelFlow converts every result into a
`Detections` object: a list of `Detection` objects, one per object found, each with the same
fields whatever model produced it.

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")
result = predictor("street.jpg")
detections = pf.from_mayaku(result, labels=predictor.class_names)
detections = detections.filter_by_confidence(0.5)

print(len(detections))
for detection in detections:
    print(detection.class_name, detection.confidence, detection.bbox)
```

```text
11
Person 0.835 [4.56, 588.91, 333.44, 1273.22]
Person 0.721 [275.62, 535.19, 469.88, 795.32]
Person 0.656 [624.09, 561.33, 743.54, 906.53]
Person 0.649 [944.12, 558.44, 1151.99, 874.55]
Umbrella 0.619 [716.73, 427.97, 957.37, 627.64]
...
```

Every function that comes after the model (filters, the tracker, zones, drawing) takes this
object, so the rest of your code does not change when you change the model.

## What a detection holds

A converter fills the fields the model provides and leaves the others empty (`None`, or an
empty list). The last fields are filled later, by the PixelFlow step named in the table.

| Field | What it holds | Filled by |
|-------|---------------|-----------|
| `bbox` | Box as `[x1, y1, x2, y2]`: the top-left and bottom-right corners, in pixels from the image's top-left corner, rounded to 2 decimals | the converter |
| `confidence` | The model's score, from 0 to 1, rounded to 3 decimals | the converter |
| `class_id` | The class number from the model | the converter |
| `class_name` | The class name, such as `"Person"`, when the model or the `labels` you pass gives one | the converter |
| `masks` | Segmentation masks, from segmentation models | the converter |
| `segments` | The object's outline as one polygon, `[[x, y], ...]` | the converter, or `masks_to_segments()` |
| `keypoints` | Landmarks as `KeyPoint` objects (`x`, `y`, `id`, `name`, `confidence`), from pose models | the converter |
| `text` | Text the model produced for this object: what an OCR model read, or a caption | the converter |
| `tracker_id` | The same number for the same object in every frame of a video | [the tracker](/docs/tracking/tracker.md) |
| `zones`, `zone_names` | The zones the object is in | [`Zones`](/docs/tracking/zones.md) |
| `line_crossings` | The lines the object crossed | [`Crossings`](/docs/tracking/crossings.md) |
| `first_seen_time`, `total_time` | When the object first appeared, and for how many seconds since | [`TimeTracker`](/docs/tracking/time-in-view.md) |
| `metadata` | Anything else, as a dictionary | you |

`class_name` and `text` are different things: `class_name` is one of the classes the model was
trained on, and `text` is free text the model produced for this one object. A detection can have
both.

## Masks and outlines

A segmentation model fills `masks` with a list holding one mask per object: a boolean array the
size of the image, `True` where the object is.

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-seg")
result = predictor("street.jpg")
detections = pf.from_mayaku(result, labels=predictor.class_names)
detections = detections.filter_by_confidence(0.5)

person = detections[0]
mask = person.masks[0]
print(mask.shape, mask.dtype, mask.sum())
```

```text
(1280, 1920) bool 117697
```

A full-size mask per object takes memory and cannot be written to JSON as it is. When you need
the outline instead, for example to store it or send it to a web page, `masks_to_segments()`
traces each mask into a polygon and stores it in `segments`:

```python
detections.masks_to_segments()
print(len(person.segments), person.segments[:3])
```

```text
105 [[15.0, 685.0], [15.0, 700.0], [19.0, 718.0]]
```

Pass `keep_masks=False` to drop the masks once they are traced, and `tolerance` (in pixels,
default 1.5) to trade outline detail for fewer points.

## Working with a Detections object

- `len(detections)` is the number of objects. An empty `Detections` is false, so
  `if not detections:` checks for "nothing found".
- `for detection in detections:` goes through them, and `detections[0]` gets one by position.
- Filters return a new `Detections` and leave the original as it is, so they chain:
  `detections.filter_by_confidence(0.5).filter_by_class_id(0)`. See
  [Filters](/docs/core-concepts/filters.md).
- `detections.copy()` returns an independent copy, for when you are about to change detections
  in place and still need the originals.
- `to_dict()` and `to_json()` turn the detections into plain data. See
  [Saving results](/docs/core-concepts/saving-results.md).

## Make detections yourself

When your model has no converter, build the detections from your own values:

```python
import pixelflow as pf

detections = pf.Detections()
detections.add_detection(
    pf.Detection(bbox=[120, 80, 260, 400], class_id=0, class_name="person", confidence=0.91)
)
detections.add_detection(
    pf.Detection(bbox=[300, 50, 300, 90], class_name="person")
)
print(len(detections))
```

```text
1
```

The second detection was not added: its box has no width. `add_detection` leaves out a
detection whose box is missing, invalid or encloses no pixels, and one whose confidence is
NaN. So every detection in a `Detections` has a usable box, and the code after it does not need
to check.

When your model returns NumPy arrays of boxes, scores and class numbers, `pf.from_arrays` builds
the whole `Detections` in one call. See [Converters](/docs/core-concepts/converters.md).
