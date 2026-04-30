"""MS COCO category ids → names, plus helpers to load ``{id: name}`` from any COCO JSON."""
import json
from pathlib import Path
from typing import Dict, Optional


def load_classes_from_coco_json(json_path: str) -> Dict[int, str]:
    """Load ``{category_id: name}`` from any COCO-format JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}


def infer_classes_from_dataset_dir(dataset_dir: str) -> Optional[Dict[int, str]]:
    """
    Auto-detect class names from a COCO-format dataset root.
    Returns ``{category_id: name}`` or ``None`` if no annotation file is found.
    """
    root = Path(dataset_dir)
    candidates = [
        root / "train" / "_annotations.coco.json",
        root / "annotations_VisDrone_train.json",
        root / "annotations_VisDrone_val.json",
        root / "annotations" / "instances_train2017.json",
        root / "annotations" / "instances_val2017.json",
        root / "val" / "_annotations.coco.json",
        root / "valid" / "_annotations.coco.json",
        root / "test" / "_annotations.coco.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                classes = load_classes_from_coco_json(str(p))
                if classes:
                    return classes
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    ann_dir = root / "annotations"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.glob("*.json")):
            try:
                classes = load_classes_from_coco_json(str(p))
                if classes:
                    return classes
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return None


def coco_classes_for_dataset(dataset_dir: Optional[str] = None) -> Dict[int, str]:
    """Prefer dataset JSON when ``dataset_dir`` is set; otherwise MS-COCO defaults."""
    if dataset_dir:
        inferred = infer_classes_from_dataset_dir(dataset_dir)
        if inferred:
            return inferred
    return COCO_CLASSES


# MS COCO 2017 val (80 classes; sparse category ids 1–90).
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}



# COCO_CLASSES = {
#     0: "person",
#     1: "bicycle",
#     2: "car",
#     3: "motorcycle",
#     4: "airplane",
#     5: "bus",
#     6: "train",
#     7: "truck",
#     8: "boat",
#     9: "traffic light",
#     10: "fire hydrant",
#     11: "stop sign",
#     12: "parking meter",
#     13: "bench",
#     14: "bird",
#     15: "cat",
#     16: "dog",
#     17: "horse",
#     18: "sheep",
#     19: "cow",
#     20: "elephant",
#     21: "bear",
#     22: "zebra",
#     23: "giraffe",
#     24: "backpack",
#     25: "umbrella",
#     26: "handbag",
#     27: "tie",
#     28: "suitcase",
#     29: "frisbee",
#     30: "skis",
#     31: "snowboard",
#     32: "sports ball",
#     33: "kite",
#     34: "baseball bat",
#     35: "baseball glove",
#     36: "skateboard",
#     37: "surfboard",
#     38: "tennis racket",
#     39: "bottle",
#     40: "wine glass",
#     41: "cup",
#     42: "fork",
#     43: "knife",
#     44: "spoon",
#     45: "bowl",
#     46: "banana",
#     47: "apple",
#     48: "sandwich",
#     49: "orange",
#     50: "broccoli",
#     51: "carrot",
#     52: "hot dog",
#     53: "pizza",
#     54: "donut",
#     55: "cake",
#     56: "chair",
#     57: "couch",
#     58: "potted plant",
#     59: "bed",
#     60: "dining table",
#     61: "toilet",
#     62: "tv",
#     63: "laptop",
#     64: "mouse",
#     65: "remote",
#     66: "keyboard",
#     67: "cell phone",
#     68: "microwave",
#     69: "oven",
#     70: "toaster",
#     71: "sink",
#     72: "refrigerator",
#     73: "book",
#     74: "clock",
#     75: "vase",
#     76: "scissors",
#     77: "teddy bear",
#     78: "hair drier",
#     79: "toothbrush",
# }


# COCO_CLASSES = {
#     1: "person",
#     2: "bicycle",
#     3: "car",
#     4: "motorcycle",
#     5: "airplane",
#     6: "bus",
#     7: "train",
#     8: "truck",
#     9: "boat",
#     10: "traffic light",
#     11: "fire hydrant",
#     12: "stop sign",
#     13: "parking meter",
#     14: "bench",
#     15: "bird",
#     16: "cat",
#     17: "dog",
#     18: "horse",
#     19: "sheep",
#     20: "cow",
#     21: "elephant",
#     22: "bear",
#     23: "zebra",
#     24: "giraffe",
#     25: "backpack",
#     26: "umbrella",
#     27: "handbag",
#     28: "tie",
#     29: "suitcase",
#     30: "frisbee",
#     31: "skis",
#     32: "snowboard",
#     33: "sports ball",
#     34: "kite",
#     35: "baseball bat",
#     36: "baseball glove",
#     37: "skateboard",
#     38: "surfboard",
#     39: "tennis racket",
#     40: "bottle",
#     41: "wine glass",
#     42: "cup",
#     43: "fork",
#     44: "knife",
#     45: "spoon",
#     46: "bowl",
#     47: "banana",
#     48: "apple",
#     49: "sandwich",
#     50: "orange",
#     51: "broccoli",
#     52: "carrot",
#     53: "hot dog",
#     54: "pizza",
#     55: "donut",
#     56: "cake",
#     57: "chair",
#     58: "couch",
#     59: "potted plant",
#     60: "bed",
#     61: "dining table",
#     62: "toilet",
#     63: "tv",
#     64: "laptop",
#     65: "mouse",
#     66: "remote",
#     67: "keyboard",
#     68: "cell phone",
#     69: "microwave",
#     70: "oven",
#     71: "toaster",
#     72: "sink",
#     73: "refrigerator",
#     74: "book",
#     75: "clock",
#     76: "vase",
#     77: "scissors",
#     78: "teddy bear",
#     79: "hair drier",
#     80: "toothbrush",
# }