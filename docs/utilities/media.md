# Read and write images, videos and cameras

<!--
summary: Read images, videos and camera streams as RGB arrays, and write them back to files, with VideoReader, VideoWriter and read_image.
label: Media
keywords: pixelflow, read image, video reader, video writer, camera stream, rgb
-->

Every image in PixelFlow is a NumPy array of shape `(height, width, 3)`, `uint8`, in RGB order.
The functions on this page read files, bytes, videos and cameras into that form, and write it
back. OpenCV, which does the decoding and encoding, uses BGR; the conversion happens here, so
the rest of your code never deals with it.

## Images

```python
import pixelflow as pf

image = pf.read_image("street.jpg")
print(image.shape, image.dtype)

pf.save_image("street.png", image)
```

```text
(1280, 1920, 3) uint8
```

`read_image` takes a path, the bytes of an encoded image, or an array:

| You pass | `read_image` |
|----------|--------------|
| a path (`str` or `Path`) | decodes the file |
| `bytes`, such as an upload or a database blob | decodes them in memory |
| a NumPy array | checks it is RGB `uint8` HxWx3 and returns it as it is |

Grayscale images come back with three channels, and an alpha channel is dropped (with a
warning). A photo whose EXIF data says it was taken rotated comes back upright, as a photo
viewer shows it.

An array is taken to be RGB already: its channel order cannot be checked. An array from
`cv2.imread` is BGR, so pass the path to `read_image` instead.

`save_image` picks the format from the extension. To send an image without writing a file, for
example as an HTTP response, `encode_image` returns the encoded bytes:

```python
data = pf.encode_image(image, ".jpg")
```

Both also write an RGBA image, shape `(height, width, 4)`, such as a cut-out from a background
removal model, and keep its transparency:

```python
pf.save_image("cutout.png", rgba)
```

Use `.png`, `.webp`, `.tif` or `.tiff`. JPEG and BMP cannot store transparency, so writing RGBA
to them raises a `ValueError` instead of choosing a background colour for you. To save the image
without its transparency, drop the alpha: `pf.save_image("cutout.jpg", rgba[:, :, :3])`. WebP
does not keep the colour of fully transparent pixels, which nobody sees anyway.

`read_image` still returns RGB, so reading the file back drops the alpha.

## Videos

`VideoReader` gives the frames of a video file one by one; `VideoWriter` writes them to a new
file:

```python
import pixelflow as pf

video = pf.VideoReader("crowd.mp4")
print(video)

with pf.VideoWriter("crowd-small.mp4", like=video) as writer:
    for frame in video:
        frame = pf.transform.resize(frame, width=640)
        writer.write(frame)
```

```text
VideoReader('crowd.mp4', 1280x720, 25.00fps, 341 frames)
```

`like=video` gives the writer the reader's frame rate. Pass `fps=` instead when there is no
reader, such as for frames you generate. The writer takes its width and height from the first
frame you write, so the loop can resize frames, as above. Every frame after that must be the same
size, or `write` raises a `ValueError`.

Use the writer in a `with` block, or call `writer.close()` at the end: the file is not complete
until it is closed.

The reader tells you what it reads:

| Attribute | |
|-----------|---|
| `width`, `height` | size of the frames, in pixels |
| `fps` | frames per second, or `None` if the file does not say |
| `frames` | number of frames, estimated from the file: good for a progress bar, not exact |
| `duration` | length in seconds |
| `codec` | the video's codec, such as `"h264"` |

To work on fewer frames, `stride=5` gives every 5th frame. `fps` and `frames` are divided to
match, so a writer made with `like=` plays the result at the right speed: for the video above,
69 frames at 5 fps instead of 341 at 25. `start=300` starts at frame 300, and `seek(0)` goes
back to the beginning; looping over the reader again does not.

`rate=2` gives 2 frames a second instead: the first frame of every half second, by the frames'
own times. Unlike a stride, it stays even in a video whose frame rate changes as it plays, which is
common in phone recordings. `fps` is the rate, so a writer made with `like=` still plays at the
right speed.

After each frame, `video.frame_number` is its number in the file, counting from 0, and
`video.frame_time` its time in milliseconds. Use them to draw a result back on the original frame,
or read it again with `read_frame`:

```python
import pixelflow as pf

with pf.VideoReader("crowd.mp4", rate=2) as video:
    for frame in video:
        print(video.frame_number, round(video.frame_time))
```

A tracker takes the frame's real time as `timestamp`, in seconds:
`tracker.update(detections, timestamp=video.frame_time / 1000)`.

`seek` is fast, but in a video whose frame rate changes as it plays, which is common in phone
recordings, it can land a few frames away from the one you asked for. When you need one exact
frame, use `read_frame`:

```python
with pf.VideoReader("crowd.mp4") as video:
    frame = video.read_frame(120)
```

It returns frame 120 of the file, counting from 0 and ignoring `stride`, and the loop continues
from the frame after it. The first time, it decodes the video up to that frame to learn each
frame's time, so reading a frame near the end of a long video takes about as long as reading the
video once. Later calls for frames before that one are fast. `video.timestamps()` gives the time of
every frame, in milliseconds.

## Cameras and streams

`CameraStream` reads a webcam by its number, or a network stream by its URL, with the same loop:

```python
import pixelflow as pf

with pf.CameraStream(0) as camera:
    with pf.VideoWriter("camera.mp4", fps=30) as writer:
        for frame in camera:
            writer.write(frame)
```

Most cameras do not report their frame rate (`camera.fps` is `None`), so give the writer
`fps=` yourself. A stream has no end and no length: the loop runs until you stop it.

## Show frames in a window

`display_video` shows one frame in a window and returns at once, so it goes inside the loop.
Pressing `q` closes the window and raises `pf.DisplayExit`, which ends the loop:

```python
import pixelflow as pf

try:
    for frame in pf.VideoReader("crowd.mp4"):
        pf.display_video(frame, width=960)
except pf.DisplayExit:
    pass
```

`width` shrinks only what is shown. `display_image(image)` shows a single image and waits for a
key; `close_display()` closes every window.

## PIL images

`to_pil(image)` turns an array into a PIL image, and `from_pil(pil_image)` turns a PIL image
into an RGB array:

```python
from PIL import Image

image = pf.from_pil(Image.open("logo.png"))
```

Whatever the PIL image's mode, the array is RGB `uint8` with three channels, as with
`read_image`: grayscale gets three channels, and an alpha channel is dropped with a warning.
