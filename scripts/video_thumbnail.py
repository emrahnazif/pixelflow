"""Make a docs thumbnail for a video: one of its frames with the play button on it.

Usage: python scripts/video_thumbnail.py VIDEO [OUTPUT] [--frame N] [--size PX]

    python scripts/video_thumbnail.py docs/assets/intro-video.mp4 docs/assets/intro-video.png

GitHub does not play videos in Markdown, so the docs show this image instead, linked to
the docs page where the video plays:

    [![Video: what it shows](/docs/assets/intro-video.png)](https://emrah.fr/pixelflow/)

OUTPUT defaults to the video's name with `.png`, next to the video: the docs site
needs the still image and the video to have the same name. The button
is scripts/play-button.png (drawn from play-button.svg), centered on the frame.
"""

import argparse
from pathlib import Path

from PIL import Image

import pixelflow as pf

BUTTON = Path(__file__).resolve().parent / "play-button.png"


def thumbnail(video: Path, frame_number: int = 0, size: int | None = None) -> Image.Image:
    """Frame *frame_number* of *video* with the play button in the middle.

    *size* is the button's width in pixels; by default a third of the frame's height,
    which makes the circle a quarter of it.
    """
    with pf.VideoReader(str(video)) as reader:
        frame = Image.fromarray(reader.read_frame(frame_number)).convert("RGBA")
    size = size or frame.height // 3
    button = Image.open(BUTTON).convert("RGBA").resize((size, size), Image.LANCZOS)
    frame.alpha_composite(button, ((frame.width - size) // 2, (frame.height - size) // 2))
    return frame.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="default: VIDEO.png")
    parser.add_argument("--frame", type=int, default=0, help="frame to use, from 0 (default 0)")
    parser.add_argument("--size", type=int, help="button width in px (default: frame height / 3)")
    args = parser.parse_args()

    output = args.output or args.video.with_suffix(".png")
    thumbnail(args.video, args.frame, args.size).save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
