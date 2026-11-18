# Converters: read results from any model

<!--
summary: Turn the output of Mayaku, Ultralytics, RF-DETR, SAM, Florence-2, EasyOCR, Detectron2 or your own model into PixelFlow Detections.
label: Converters
keywords: pixelflow, converters, mayaku, ultralytics, rf-detr, segment anything
-->

A converter takes a model's output, exactly as the model returns it, and gives back a
[`Detections`](/docs/core-concepts/detections.md) object. It is the only line that depends on
the model: everything after it works the same whichever converter you used.

| Model or library | Converter |
|------------------|-----------|
| [Mayaku](https://emrah.fr/mayaku/) | `pf.from_mayaku` |
| Ultralytics (YOLO detection, segmentation, pose) | `pf.from_ultralytics` |
| RF-DETR | `pf.from_rfdetr` |
| supervision (`sv.Detections`) | `pf.from_supervision` |
| Segment Anything (SAM, SAM 2) | `pf.from_sam` |
| EfficientTAM | `pf.from_efficienttam` |
| Florence-2 | `pf.from_florence2` |
| EasyOCR | `pf.from_easyocr` |
| Detectron2 | `pf.from_detectron2` |
| Falcon Perception | `pf.from_falcon_perception` |
| Any model that returns arrays | `pf.from_arrays` |

For classification models, see [Classifications](/docs/core-concepts/classifications.md).

## Class names: the `labels` argument

Models return class numbers. To get class names in `class_name`, most converters take
`labels`, in one of three forms:

- a list, where the position is the class number: `["person", "bicycle", "car"]`
- a dictionary from class number to name: `{1: "person", 2: "bicycle"}`
- a list of dictionaries, which can also name keypoints:
  `[{"id": 0, "name": "person", "keypoints": ["nose", "left_eye", ...]}]`

Take the names from the model itself, as the examples below do, rather than typing a list: a
list that does not match the model's numbering gives every detection the wrong name, with no
error. A class number that has no name in `labels` is kept, and `class_name` stays `None`.

## Mayaku

```python
import pixelflow as pf
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")
result = predictor("street.jpg")
detections = pf.from_mayaku(result, labels=predictor.class_names)
```

Fills boxes, scores and class numbers, and masks with a segmentation model such as
`mayaku-n-seg`. `predictor.class_names` holds the names the model was trained with. For several
images, `predictor.batch` returns one result per image: convert each one separately.

## Ultralytics

```python
import pixelflow as pf
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
results = model.predict("street.jpg", verbose=False)
detections = pf.from_ultralytics(results)
```

Works with detection, segmentation and pose models. Ultralytics stores class names in the model,
so `labels` is only needed to rename them. A segmentation model fills both `masks` and
`segments`.

## RF-DETR and supervision

```python
import pixelflow as pf
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRNano()
predictions = model.predict("street.jpg", threshold=0.5)
detections = pf.from_rfdetr(predictions, labels=COCO_CLASSES)
```

> **Note:** RF-DETR numbers the COCO classes from 1 (`1` is person), but `model.class_names`
> is a list that starts at 0. Passing that list shifts every name by one. `COCO_CLASSES` is a
> dictionary with RF-DETR's own numbers, so the names come out right. For a model you trained
> on your own classes, check the numbering the same way.

RF-DETR returns an `sv.Detections` from the supervision library, so `pf.from_rfdetr` is the
same function as `pf.from_supervision`: use either name for any `sv.Detections`.

## Segment Anything

SAM and SAM 2 return candidate masks for the point or box you give them, each with a quality
score. `pf.from_sam` turns each candidate into a detection, with the box taken from the mask:

```python
import numpy as np
import pixelflow as pf
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny", device="cpu")
image = pf.read_image("street.jpg")
predictor.set_image(image)

masks, scores, _ = predictor.predict(
    point_coords=np.array([[170, 900]]), point_labels=np.array([1])
)
detections = pf.from_sam(masks, scores)

for detection in detections:
    print(detection.confidence, detection.bbox)
```

```text
0.446 [8.0, 599.0, 365.0, 1279.0]
0.459 [55.0, 762.0, 319.0, 1108.0]
0.883 [29.0, 718.0, 322.0, 1107.0]
```

Use `device="cuda"` on an NVIDIA GPU. SAM does not name what it segments, so `class_id` and
`class_name` are `None`. EfficientTAM returns the same format: use `pf.from_efficienttam`.

## Florence-2

Florence-2 does several tasks, chosen by a prompt such as `<OD>` (object detection). Pass the
same prompt to the converter:

```python
import pixelflow as pf
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

model_id = "florence-community/Florence-2-base"
model = Florence2ForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("street.jpg")
task = "<OD>"
inputs = processor(text=task, images=image, return_tensors="pt")
generated = model.generate(**inputs, max_new_tokens=1024)
text = processor.batch_decode(generated, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(text, task=task, image_size=image.size)

detections = pf.from_florence2(parsed, task_prompt=task, image_size=image.size)
```

Object detection tasks fill `class_name`. Captioning, grounding and OCR tasks with regions put
the text the model produced in `text`. Segmentation tasks fill `segments`. Text-only tasks, which
return no regions, raise a `ValueError`.

## EasyOCR

```python
import easyocr
import pixelflow as pf

reader = easyocr.Reader(["en"])
detections = pf.from_easyocr(reader.readtext("street.jpg"))
```

One detection per text region: the text read in `text`, the four corners of the region in
`segments`, and the box around them in `bbox`. Call `readtext` with its default `detail=1`:
with `detail=0` EasyOCR returns only the strings, which have no position to convert.

## Detectron2

```python
import pixelflow as pf
from detectron2.data import MetadataCatalog

outputs = predictor(image)  # a detectron2 DefaultPredictor, BGR image
class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
detections = pf.from_detectron2(outputs, labels=class_names)
```

Fills boxes, scores, class numbers, masks and keypoints.

## Falcon Perception

Falcon Perception finds what you name in a query, so pass the query as `label` to use it as the
class name, and the image size, because the model returns coordinates relative to the image:

```python
detections = pf.from_falcon_perception(
    output, image_size=(image.width, image.height), label="cat"
)
```

`output` is what the local engine's `generate` returns, or the JSON from the Falcon Perception
server.

## Any other model: `from_arrays`

When your model returns NumPy arrays or PyTorch tensors instead of a framework object, pass them
to `pf.from_arrays`:

```python
import numpy as np
import pixelflow as pf

boxes = np.array([[120, 80, 260, 400], [300, 120, 520, 330]])
scores = np.array([0.91, 0.78])
class_ids = np.array([0, 2])

detections = pf.from_arrays(boxes, scores, class_ids, labels=["person", "bicycle", "car"])

for detection in detections:
    print(detection.class_name, detection.confidence, detection.bbox)
```

```text
person 0.91 [120.0, 80.0, 260.0, 400.0]
car 0.78 [300.0, 120.0, 520.0, 330.0]
```

| Argument | Shape | |
|----------|-------|---|
| `boxes` | `(N, 4)` | `[x1, y1, x2, y2]` in pixels |
| `scores` | `(N,)` | confidence |
| `class_ids` | `(N,)` | optional: leave it out for a model that finds things without naming them |
| `masks` | `(N, H, W)` | optional, turned into boolean masks |
| `keypoints` | `(N, K, 3)` | optional, `(x, y, score)` per keypoint |
| `segments` | `N` polygons | optional |
| `texts` | `(N,)` | optional, text per detection |
| `labels` | | optional class names, as above |

Tensors on a GPU are moved to the CPU for you.
