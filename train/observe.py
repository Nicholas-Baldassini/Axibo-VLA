"""
The policy's image: one or more recorded cameras cropped and tiled into a single frame, plus the
train-time augmentation and the matching eval-time crop.

OpenVLA takes exactly one 224x224 image, and its processor squashes whatever it is given to that
size without cropping (image_resize_strategy "resize-naive" in the checkpoint), so the canvas can
be any shape: two 640x480 views side by side become two 112x224 halves. observation.image in the
training config names the cameras:

    "image": "observation.images.front"                                   one camera, as before
    "image": ["observation.images.front", "observation.images.wrist"]     tiled left to right
    "crops": {"observation.images.front": [40, 480, 0, 640]}              rows/cols kept per camera
    "augment": True                                                        OpenVLA's fine-tuning augmentation

Augmentation follows OpenVLA's own fine-tuning pipeline (prismatic/vla/datasets/datasets.py,
image_aug=True): after the resize to 224, a random crop covering 90% of the area, resized back to
224, then brightness 0.2, contrast 0.8-1.2, saturation 0.8-1.2, hue 0.05. At inference OpenVLA
centre-crops the same 90% (its evaluation's --center_crop, "important because we fine-tuned with
random crop augmentations"), which is what eval_view does. Without "augment" the raw canvas goes
to the processor, as every run before 2026-09-29 did.

Why: measured on run 7, the policy reproduces the demonstrations' actions to 0.3 mm per step on
recorded frames but drifts in closed loop, the signature of a model tied to exact pixels; our
renders carry no visual noise at all. Random crops are the standard remedy in behaviour cloning
and the one part of OpenVLA's recipe this pipeline had left out.
"""

import numpy as np
from PIL import Image

MODEL_SIZE = 224
CROP_AREA = 0.9          # fraction of the area kept by the random / centre crop


def spec(observation_cfg):
    """Normalized observation spec from a training config's "observation" section."""
    image = observation_cfg["image"]
    keys = [image] if isinstance(image, str) else list(image)
    return {
        "keys": keys,
        "crops": {k: tuple(int(v) for v in box) for k, box in (observation_cfg.get("crops") or {}).items()},
        "augment": bool(observation_cfg.get("augment", False)),
    }


def camera_names(spec_):
    return [k.split("observation.images.", 1)[-1] for k in spec_["keys"]]


def describe(spec_):
    views = " + ".join(camera_names(spec_))
    crops = ", ".join(f"{k.split('.')[-1]} rows {t}-{b} cols {l}-{r}" for k, (t, b, l, r) in spec_["crops"].items())
    return views + (f" (crops: {crops})" if crops else "") + (" with augmentation" if spec_["augment"] else "")


def compose(images, spec_):
    """One uint8 canvas (H, W, 3) from {camera key or name: uint8 (h, w, 3)}: each view cropped to its
    box, scaled to a common height, tiled left to right. Width is kept even for video encoders."""
    views = []
    for key in spec_["keys"]:
        img = images[key] if key in images else images[key.split("observation.images.", 1)[-1]]
        if key in spec_["crops"]:
            t, b, l, r = spec_["crops"][key]
            img = img[t:b, l:r]
        views.append(np.ascontiguousarray(img))
    if len(views) == 1:
        return views[0]
    height = max(v.shape[0] for v in views)
    scaled = []
    for v in views:
        if v.shape[0] != height:
            w = int(round(v.shape[1] * height / v.shape[0]))
            v = np.asarray(Image.fromarray(v).resize((w, height), Image.BILINEAR))
        scaled.append(v)
    canvas = np.concatenate(scaled, axis=1)
    if canvas.shape[1] % 2:
        canvas = canvas[:, :-1]
    return canvas


def _to_model_size(img):
    return Image.fromarray(img).resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)


def train_view(canvas, spec_, rng):
    """PIL image for training: the canvas at 224x224, augmented when the spec says so."""
    img = _to_model_size(canvas)
    if not spec_["augment"]:
        return img
    import torchvision.transforms as T
    side = int(round(MODEL_SIZE * np.sqrt(CROP_AREA)))
    top, left = rng.integers(0, MODEL_SIZE - side + 1, size=2)
    img = img.crop((int(left), int(top), int(left) + side, int(top) + side)).resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)
    jitter = T.ColorJitter(brightness=0.2, contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05)
    return jitter(img)


def eval_view(canvas, spec_):
    """PIL image for inference: the canvas at 224x224, centre-cropped to the augmentation's area
    when the run was trained with augmentation, so the policy sees the crop scale it was trained on."""
    img = _to_model_size(canvas)
    if not spec_["augment"]:
        return img
    side = int(round(MODEL_SIZE * np.sqrt(CROP_AREA)))
    off = (MODEL_SIZE - side) // 2
    return img.crop((off, off, off + side, off + side)).resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)
