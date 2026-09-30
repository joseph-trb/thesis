import os
import cv2
import json
import hashlib
import tarfile
import numpy as np
import xml.etree.ElementTree as ET

MODEL_RATIO = 25
MODEL_NAME = "resnet"
VERSION = "v4"
MODEL_PATH = f"/work/dldevel/trieba/bachelor_thesis/results/model_{MODEL_RATIO}/{MODEL_NAME}/{VERSION}"

OUTPUT_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/eval_dataset"
BBOX_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/ImageNet_BoundingBoxes/Annotations"

_bbox_cache = {}


def get_bboxes_for_class(tar_path):
    bboxes = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".xml"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                tree = ET.parse(f)
                root = tree.getroot()

                filename = root.find('filename').text

                size_elem = root.find('size')
                img_w = img_h = None
                if size_elem is not None:
                    w_elem, h_elem = size_elem.find('width'), size_elem.find('height')
                    if w_elem is not None and h_elem is not None:
                        img_w, img_h = int(w_elem.text), int(h_elem.text)

                bboxes[filename] = []

                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue

                    xmin = max(0, int(float(bndbox.find('xmin').text)))
                    ymin = max(0, int(float(bndbox.find('ymin').text)))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    if img_w is not None:
                        xmax = min(xmax, img_w)
                    if img_h is not None:
                        ymax = min(ymax, img_h)

                    if xmax > xmin and ymax > ymin:
                        bboxes[filename].append([xmin, ymin, xmax, ymax])
    return bboxes


def get_bboxes_cached(synset):
    if synset not in _bbox_cache:
        tar_path = os.path.join(BBOX_PATH, f"{synset}.tar.gz")
        _bbox_cache[synset] = get_bboxes_for_class(tar_path) if os.path.exists(tar_path) else {}
    return _bbox_cache[synset]


def clamp_boxes(bbox_list, h, w):
    clamped = []
    for x_min, y_min, x_max, y_max in bbox_list:
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        if x_max > x_min and y_max > y_min:
            clamped.append([x_min, y_min, x_max, y_max])
    return clamped


def seed_from_stem(stem):
    """Deterministic, per-image seed. Fixes the original bug where every
    image reused the same hardcoded seed=0, which made the patch search
    probe the exact same spatial locations for every image in the dataset
    regardless of content."""
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16) % (2**32)


def largest_rectangle_in_histogram(heights):
    """Returns (area, left, right_inclusive, height)."""
    stack = []
    best_area, best = 0, (0, 0, 0, 0)
    n = len(heights)
    for i in range(n + 1):
        h = heights[i] if i < n else 0
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            height = heights[top]
            left = stack[-1] + 1 if stack else 0
            width = i - left
            area = height * width
            if area > best_area:
                best_area = area
                best = (area, left, i - 1, height)
        stack.append(i)
    return best


def largest_clean_rectangle(mask_2d):
    """Largest axis-aligned rectangle where mask_2d == 0 (i.e. NOT inside
    any bounding box). Returns ((y0, x0, y1, x1) exclusive bounds, area)."""
    h, w = mask_2d.shape
    heights = [0] * w
    best_area, best_rect = 0, (0, 0, 1, 1)
    for y in range(h):
        row = mask_2d[y]
        for x in range(w):
            heights[x] = heights[x] + 1 if row[x] == 0 else 0
        area, left, right, height = largest_rectangle_in_histogram(heights)
        if area > best_area:
            best_area = area
            best_rect = (y - height + 1, left, y + 1, right + 1)
    return best_rect, best_area


def fill_bbox_from_largest_rect(image, bbox_mask_2d, bbox_list, seed):
    (ry0, rx0, ry1, rx1), area = largest_clean_rectangle(bbox_mask_2d)
    out = image.copy()

    if area == 0:
        mean_color = image.reshape(-1, 3).mean(axis=0).astype(image.dtype)
        source_patch = np.tile(mean_color, (32, 32, 1))
    else:
        source_patch = image[ry0:ry1, rx0:rx1]

    ph, pw = source_patch.shape[:2]
    rng = np.random.default_rng(seed)

    for x_min, y_min, x_max, y_max in bbox_list:
        box_h, box_w = y_max - y_min, x_max - x_min
        if box_h <= 0 or box_w <= 0:
            continue

        if ph >= box_h and pw >= box_w:
            oy = int(rng.integers(0, ph - box_h + 1))
            ox = int(rng.integers(0, pw - box_w + 1))
            fill = source_patch[oy:oy + box_h, ox:ox + box_w]
        else:
            reps_y = int(np.ceil(box_h / ph))
            reps_x = int(np.ceil(box_w / pw))
            fill = np.tile(source_patch, (reps_y, reps_x, 1))[:box_h, :box_w]

        out[y_min:y_max, x_min:x_max] = fill

    return out


with open(f"{MODEL_PATH}/synset_to_idx.json", "r") as f:
    synset_to_idx = json.load(f)
with open(f"{MODEL_PATH}/maps_to_idx.json", "r") as f:
    maps_to_idx = json.load(f)
idx_to_maps = {k: v for v, k in maps_to_idx.items()}

cls_to_synset = {idx_to_maps[int(idx)]: synset for synset, idx in synset_to_idx.items()}

# regenerate only_bg_t for every image already in original/
original_root = os.path.join(OUTPUT_PATH, "original")
target_root = os.path.join(OUTPUT_PATH, "only_bg_t")

fixed, skipped = 0, 0

for cls in sorted(os.listdir(original_root)):
    synset = cls_to_synset.get(cls)
    if synset is None:
        print(f"Warning: no synset found for class folder '{cls}'. Skipping.")
        continue

    class_bboxes = get_bboxes_cached(synset)
    cls_dir = os.path.join(original_root, cls)
    out_dir = os.path.join(target_root, cls)
    os.makedirs(out_dir, exist_ok=True)

    for img_name in sorted(os.listdir(cls_dir)):
        img_stem = img_name.split(".")[0]

        bbox_list = class_bboxes.get(img_stem)
        if not bbox_list:
            print(f"Warning: no bbox for {cls}/{img_name}. Leaving existing file untouched.")
            skipped += 1
            continue

        img_bgr = cv2.imread(os.path.join(cls_dir, img_name))
        if img_bgr is None:
            print(f"Warning: could not read {cls}/{img_name}. Skipping.")
            skipped += 1
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        clamped_boxes = clamp_boxes(bbox_list, h, w)
        if not clamped_boxes:
            print(f"Warning: no usable bbox after clamping for {cls}/{img_name}. Skipping.")
            skipped += 1
            continue

        bbox_mask_2d = np.zeros((h, w), dtype=np.uint8)
        for x_min, y_min, x_max, y_max in clamped_boxes:
            bbox_mask_2d[y_min:y_max, x_min:x_max] = 1

        seed = seed_from_stem(img_stem)
        fixed_img = fill_bbox_from_largest_rect(img, bbox_mask_2d, clamped_boxes, seed)

        cv2.imwrite(os.path.join(out_dir, img_name), cv2.cvtColor(fixed_img, cv2.COLOR_RGB2BGR))
        fixed += 1

print(f"\nDone. Regenerated {fixed} only_bg_t images, skipped {skipped}.")