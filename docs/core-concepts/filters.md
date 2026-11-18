# Filters: keep only the detections you need

<!--
summary: Keep detections by confidence, class, size, shape, position or tracking time, and remove duplicates, by chaining filter methods.
label: Filters
keywords: pixelflow, filter detections, confidence threshold, class filter, remove duplicates
-->

A model returns everything it might have seen, including guesses with a low score and classes
you do not care about. Filters keep the detections you need. Each one is a method of
[`Detections`](/docs/core-concepts/detections.md) that returns a new `Detections`, so they
chain, and the original stays as it was.

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")
result = predictor("street.jpg")
detections = pf.from_mayaku(result, labels=predictor.class_names)

person = predictor.class_names.index("Person")
people = detections.filter_by_confidence(0.5).filter_by_class_id(person)

print(len(detections), len(people))
```

```text
279 8
```

The model returned 279 detections; 8 are people with a confidence of 0.5 or more.

## Confidence and class

| Filter | Keeps |
|--------|-------|
| `filter_by_confidence(threshold)` | detections with a confidence of `threshold` or more |
| `filter_by_class_id(class_ids)` | detections whose `class_id` is in `class_ids`: one id or a list |
| `remap_class_ids(from_ids, to_id)` | every detection, with the ids in `from_ids` changed to `to_id` |

`filter_by_class_id` compares class numbers, not names. Get the numbers from the model's class
names, as above, rather than writing them by hand: the same number means different classes in
different models.

`remap_class_ids` merges several classes into one, for example to count every two-wheeler
together:

```python
two_wheelers = [predictor.class_names.index(name) for name in ("Motorcycle", "Scooter", "Bicycle")]
vehicles = detections.filter_by_confidence(0.3).filter_by_class_id(two_wheelers)
vehicles = vehicles.remap_class_ids(two_wheelers, two_wheelers[0])
```

The remapped detections have `class_name` set to `None`, because the old name no longer
matches the new number.

## Size and shape

| Filter | Keeps detections whose box |
|--------|----------------------------|
| `filter_by_size(min_area=None, max_area=None)` | has an area, in pixels, within the range |
| `filter_by_dimensions(min_width=None, max_width=None, min_height=None, max_height=None)` | has a width and height, in pixels, within the ranges |
| `filter_by_aspect_ratio(min_ratio=None, max_ratio=None)` | has a width ÷ height within the range |
| `filter_by_relative_size(min_percent=None, max_percent=None, frame_width=None, frame_height=None)` | covers a share of the image within the range, from 0 to 1 |

Leave out a limit to keep everything on that side. All limits are inclusive.

Pixel sizes depend on the image size: in the same photo at half the width and height, every box
covers a quarter of the pixels. When your images come in different sizes,
`filter_by_relative_size` keeps the same objects in all of them:

```python
# people covering at least 2% of a 1920x1280 image
close = people.filter_by_relative_size(min_percent=0.02, frame_width=1920, frame_height=1280)
```

An aspect ratio below 1 means taller than wide: `max_ratio=0.8` keeps standing people and drops
boxes that are wider than they are tall.

## Position in the image

`filter_by_position` keeps detections by where the center of their box is:

```python
left = people.filter_by_position("left", margin_percent=0.25, frame_width=1920, frame_height=1280)
```

| `region` | Keeps detections whose center is |
|----------|----------------------------------|
| `"left"`, `"right"`, `"top"`, `"bottom"` | within `margin_percent` of that side (`0.25` = the outer quarter) |
| `"edge"` | within `margin_percent` of any side |
| `"corners"` | within `margin_percent` of two sides at once |
| `"center"` | further than `margin_percent` from every side |

`margin_percent` defaults to `0.1`. `frame_width` and `frame_height` are required.

For any other area, such as a doorway or a lane, draw it as a zone:
`filter_by_zones(zone_ids, exclude=False)` keeps the detections inside the zones you name, or
outside them with `exclude=True`. It reads what `Zones` recorded, so run the zones first. See
[Count objects in a zone](/docs/tracking/zones.md).

## Duplicates and overlaps

| Filter | Keeps |
|--------|-------|
| `remove_duplicates(iou_threshold=0.8, keep="first")` | one detection from each group of boxes that overlap by `iou_threshold` or more |
| `filter_overlapping(min_overlap=0.5, target_class_ids=None)` | detections that overlap another detection by `min_overlap` or more |

Overlap is measured as IoU: the area two boxes share divided by the area they cover together,
from 0 (apart) to 1 (the same box).

Some models, or two models whose results you combine, give the same object twice.
`remove_duplicates` keeps one from each group: `keep="first"`, `"last"` or
`"highest_confidence"`.

`filter_overlapping` finds objects that appear together. With `target_class_ids`, it only
counts overlaps with those classes: for example, the detections that overlap a person.

## Tracked objects

After [the tracker](/docs/tracking/tracker.md) and [`TimeTracker`](/docs/tracking/time-in-view.md)
have run on a video, three more filters use what they recorded:

| Filter | Keeps |
|--------|-------|
| `filter_tracked_objects(require_tracker_id=True)` | detections the tracker gave an id, or with `False`, the ones it did not |
| `filter_by_tracking_duration(min_seconds=None, max_seconds=None)` | objects seen for a time within the range |
| `filter_by_first_seen_time(start_time=None, end_time=None)` | objects first seen within the time range |

`filter_by_tracking_duration(min_seconds=2)` is a simple way to ignore objects that appear for a
moment: a false detection rarely lasts two seconds.
