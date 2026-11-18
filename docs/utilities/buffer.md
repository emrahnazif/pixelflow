# Look at past and future frames with a buffer

<!--
summary: Hold the last few frames of a video with their results, so each frame can be processed knowing the frames before and after it, with pf.Buffer.
label: Buffer
keywords: pixelflow, frame buffer, temporal context, video delay, moving average
-->

Some decisions about a frame are easier with the frames around it: a count that jumps from one
frame to the next, a detection that appears for a single frame. `pf.Buffer` keeps the last few
frames with their results and hands you the one in the middle, a few frames late. For that
frame, you have both the frames before it and the frames after it.

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")
person = predictor.class_names.index("Person")
buffer = pf.Buffer(frames=5)

video = pf.VideoReader("crowd.mp4")
with pf.VideoWriter("crowd-buffered.mp4", like=video) as writer:
    for frame in video:
        result = predictor(frame[:, :, ::-1])  # Mayaku reads arrays as BGR
        detections = pf.from_mayaku(result, labels=predictor.class_names)
        people = detections.filter_by_confidence(0.5).filter_by_class_id(person)

        people, frame = buffer.update(people, frame)
        if not buffer.is_full:
            continue

        context = buffer.get_temporal_context()
        window = context["past_results"] + [people] + context["future_results"]
        average = sum(len(d) for d in window) / len(window)
        if buffer.frames_processed % 50 == 0:
            print(f"{len(people)} people in this frame, {average:.1f} on average")

        frame = pf.annotate.box(frame, people)
        writer.write(frame)
```

```text
21 people in this frame, 22.6 on average
29 people in this frame, 29.2 on average
23 people in this frame, 23.0 on average
27 people in this frame, 27.6 on average
21 people in this frame, 21.6 on average
16 people in this frame, 16.4 on average
```

![A frame of a crowded hall seen from above, with a box around each person](/docs/assets/buffer.jpg)

`buffer.update(results, frame)` stores the new frame and its results, and returns the frame and
results from the middle of the buffer: with `frames=5`, the ones from 2 frames ago. They always
belong together, so the boxes you draw are the right ones for the frame you draw on.

`get_temporal_context()` returns the whole buffer around that middle frame. Above, the count of
each frame is averaged with the 2 frames before and the 2 after. Because the window is centered
on the frame, the average does not lag behind the crowd, as an average of past frames only
would.

## The first and last frames

Until the buffer is full, `update` returns a black frame and empty `Detections`: with
`frames=5`, for the first 4 calls. Skip them with `if not buffer.is_full: continue`, as above,
or the output video starts with black frames.

The buffer does not return the first 2 and the last 2 frames of the video: the first ones were
only ever "past" for frame 2, and the last ones are still in the buffer when the loop ends.
Above, the output video has 337 of the 341 frames.

`buffer.delay` is how many frames late the output is: `frames // 2`. A larger buffer gives more
context and a longer delay. Use an odd number of frames, so that there is one frame in the
middle.

## The context

`get_temporal_context()` returns `None` until the buffer is full, then a dictionary:

| Key | |
|-----|---|
| `past_frames`, `past_results` | lists of the frames and results before the middle one, oldest first |
| `current_frame`, `current_results` | the middle frame and results, the ones `update` returned |
| `future_frames`, `future_results` | lists of the frames and results after it, oldest first |

The results can be anything you pass to `update`: `Detections`, a count or a dictionary of your
own.

`get_buffer_contents()` returns `(results, frames)` as two lists, oldest first. `reset()` empties
the buffer, for example before the next video.

## Parameters

`pf.Buffer(frames=5)`:

| Parameter | Default | |
|-----------|---------|---|
| `frames` | `5` | number of frames to keep, from 1 to 1000; an even number gives a warning |

| Attribute | |
|-----------|---|
| `is_full` | whether the buffer holds `frames` frames |
| `delay` | how many frames late `update` returns: `frames // 2` |
| `frames_processed` | number of `update` calls since the start or the last `reset()` |
| `current_size` | number of frames in the buffer now |

Every frame is copied when it enters the buffer: 5 frames of 1280×720 use 14 MB.
