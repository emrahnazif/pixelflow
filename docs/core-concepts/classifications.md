# Classifications: what a model says about an image

<!--
summary: Read a classifier's answers as a Classifications object: take the best one, the top few, or every class above a score.
label: Classifications
keywords: pixelflow, image classification, top-k, class scores, zero-shot
-->

A classification model does not find objects. It gives a score to every class it knows, for
the whole image. PixelFlow turns those scores into a `Classifications` object: one
`Classification` per class, each with a `class_id`, a `class_name` and a `confidence`.

The example below uses torchvision's ResNet-18, because Mayaku does not publish a
classification model.

```python
import torch
import pixelflow as pf
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
preprocess = weights.transforms()

image = pf.read_image("street.jpg")
batch = preprocess(pf.to_pil(image)).unsqueeze(0)
with torch.no_grad():
    probabilities = model(batch).softmax(dim=-1)[0]

result = pf.from_scores(probabilities, labels=weights.meta["categories"])

print(len(result))
print(result.top1)
for answer in result.top_k(3):
    print(answer.class_name, answer.confidence)
```

```text
1000
Classification(class_id=670, class_name='motor scooter', confidence=0.315)
motor scooter 0.315
golfcart 0.28
tricycle 0.073
```

The model knows 1,000 classes, so the result has 1,000 rows. `top1` is the best answer and
`top_k(3)` the three best.

## Detections or classifications

PixelFlow keeps the two apart because they mean different things:

- Three **detections** are three objects in the image, each with its own box.
- Three **classifications** are three competing answers about one image. None has a box,
  because the model did not say where anything is.

So a `Classifications` object has no boxes, and the functions that need boxes (the tracker,
zones, box drawing) do not take it. To draw the answers on the image, use
[`pf.annotate.classification`](/docs/visualization/classification.md).

## Rank or threshold

Which method to use depends on what the scores mean:

- **One right answer** (a softmax over all classes, as above): rank them. `result.top1` is
  the best answer, `result.top_k(5)` the five best, sorted best first.
- **Any number of right answers** (independent scores, such as a multi-label model or CLIP
  similarities against several prompts): keep every class above a score with
  `result.filter_by_confidence(0.3)`. The number of answers can change from image to image,
  which a fixed `top_k` cannot follow.

`top_k`, `filter_by_confidence` and slicing (`result[:3]`) all return a new
`Classifications`. `len()`, `for answer in result:` and `result[0]` work as on
[Detections](/docs/core-concepts/detections.md).

## Scores are kept as given

PixelFlow does not apply softmax or change the scores in any way: `confidence` is exactly the
number you passed. A softmax probability, a sigmoid score and a CLIP similarity are all valid,
and only you know which one your model gives. If your model returns raw logits, convert them
first: `logits.softmax(dim=-1)` when there is one right answer, `logits.sigmoid()` when there
can be several.

`from_scores` takes a PyTorch tensor, a NumPy array or a list. `labels` gives the class names,
as a list (position = class number), a dictionary (`{0: "cat", 1: "dog"}`) or a list of
`{"id": ..., "name": ...}` dictionaries. A class with no name in `labels` keeps its number and
has `class_name` set to `None`.

## Other ways to get classifications

- **Ultralytics classification models** (`yolo11n-cls.pt` and other `-cls` checkpoints):
  `pf.from_ultralytics_classification(model.predict("street.jpg"))`. Pass `top_k=5` to keep
  only the five best on every video frame, instead of 1,000 rows each time.
- **Your own values:**

  ```python
  import pixelflow as pf

  result = pf.from_scores([0.62, 0.31, 0.07], labels=["street", "market", "parking lot"])
  print(result.top1.class_name)
  ```

  ```text
  street
  ```

To store or send the result, `to_dict()` and `to_json()` work as for detections. See
[Saving results](/docs/core-concepts/saving-results.md).
