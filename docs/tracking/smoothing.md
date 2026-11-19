# Smooth jittery boxes

<!--
summary: Steady the boxes of tracked objects by averaging each one over the frames around it, and fill in boxes the model missed, with pf.smooth.
label: Smoothing
keywords: pixelflow, smooth bounding boxes, jitter, temporal smoothing, interpolation
-->

A detection model draws each frame's boxes from scratch, so a person walking at a steady pace gets
a box that shakes by a pixel or two from frame to frame, and now and then a frame with no box at
all. On a video the boxes look nervous, and speeds or directions measured from them are noisy.
`pf.smooth` averages each tracked object's box over the frames around it, using a
[buffer](/docs/utilities/buffer.md) that holds those frames.

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")
person = predictor.class_names.index("Person")

video = pf.VideoReader("crowd.mp4")
tracker = pf.tracker.ByteTrack(frame_rate=video.fps)
buffer = pf.Buffer(frames=5)

with pf.VideoWriter("crowd-smooth.mp4", like=video) as writer:
    for frame in video:
        result = predictor(frame[:, :, ::-1], conf=0.1)  # Mayaku reads arrays as BGR
        detections = pf.from_mayaku(result, labels=predictor.class_names)
        people = tracker.update(detections.filter_by_class_id(person)).filter_tracked_objects()

        people, frame = buffer.update(people, frame)
        if frame is not None:
            writer.write(pf.annotate.box(frame, pf.smooth(buffer)))

    for people, frame in buffer.flush():
        writer.write(pf.annotate.box(frame, pf.smooth(buffer)))

print(writer.frames_written, "frames")
```

```text
341 frames
```

![Four seconds of the crowd video, zoomed in, side by side: on the left the boxes from the model shake from frame to frame, on the right the smoothed boxes follow each person steadily](/docs/assets/smoothing.mp4)

The buffer hands each frame back 2 frames late, once the 2 frames after it have arrived;
`pf.smooth(buffer)` then returns that frame's detections, each box averaged with the same
object's boxes in the 2 frames before and the 2 after. Objects are matched by `tracker_id`, so
smoothing needs a [tracker](/docs/tracking/tracker.md): detections without an id are left out of
the result, and if none of a frame's detections has one, `smooth` warns and returns them as they
are. Class, id, masks and keypoints come from the frame itself; only the box and the confidence
are averaged.

Near the start and the end of the video, a frame does not have 2 frames on each side: `smooth`
returns its detections as they are.

## How much smoother

The weight of a frame falls with its distance from the frame being smoothed:
`temporal_weight_decay` (0.8) for the next frame on each side, 0.8 × 0.8 for the one after, and
so on. A smaller value keeps the box closer to the frame's own; a larger buffer averages over more
frames.

On the example, measured as how much a box's movement changes from one frame to the next (the
mean absolute second difference, in pixels):

| | Box center x | Box center y | Width | Height | Filled in | Delay |
|-|--------------|--------------|-------|--------|-----------|-------|
| Boxes from the model | 0.95 | 1.39 | 1.79 | 2.35 | | |
| `Buffer(frames=5)` | 0.21 | 0.38 | 0.40 | 0.58 | 18 | 2 frames |
| `Buffer(frames=5)`, `temporal_weight_decay=0.5` | 0.27 | 0.48 | 0.52 | 0.75 | 18 | 2 frames |
| `Buffer(frames=9)` | 0.14 | 0.26 | 0.26 | 0.37 | 43 | 4 frames |

The cost is delay, 80 ms at 25 fps for 5 frames. Boxes that follow fast changes of direction
need a small buffer; slow scenes can take a larger one.

## Boxes filled in

When an object has no box in a frame but has one in the frames before and after it, `smooth` adds
one: halfway between its nearest box before and its nearest box after, with 85% of their average
confidence. Those detections carry `metadata={"interpolated": True}`:

```python
filled = [d for d in pf.smooth(buffer) if d.metadata and d.metadata.get("interpolated")]
```

## Parameters

`pf.smooth(...)`:

| Parameter | Default | |
|-----------|---------|---|
| `buffer` | | the `pf.Buffer` holding tracked detections; `update` or `flush` must have returned a frame |
| `temporal_weight_decay` | `0.8` | weight of each step away from the smoothed frame, from 0.1 to 1 |

When it smooths, it returns a new `Detections` and leaves the buffer unchanged. See
[look at past and future frames with a buffer](/docs/utilities/buffer.md) for `Buffer`'s own
options.
