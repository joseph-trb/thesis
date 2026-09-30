import os
import cv2
import json
import torch
import random
import tarfile
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# val_bbox_only_manifest.json, synset_to_idx.json, maps_to_idx.json are identical for all models
MODEL_RATIO = 25
MODEL_NAME = "resnet"
VERSION = "v4"
MODEL_PATH = f"/work/dldevel/trieba/bachelor_thesis/results/model_{MODEL_RATIO}/{MODEL_NAME}/{VERSION}"

TARGET_IMAGES_PER_CLASS = 100
MAX_DONOR_ATTEMPTS = 20

IMAGENET_PATH = "/work/dldevel/galella/datasets/ImageNet/ILSVRC2012_img_train"
OUTPUT_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/eval_dataset"
MASK_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/SAM/ImageNet_validation_masks"
BBOX_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/ImageNet_BoundingBoxes/Annotations"
SUPERCLASS_MAP = "superclasses_mapping.json"
CHECKPOINT_SAM2 = "/work/dldevel/trieba/bachelor_thesis/training_data/SAM/checkpoint/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

VARIANTS = [
    "original", "only_bg_b", "only_bg_t", "only_bg_noise",
    "no_fg", "no_fg_noise", "only_fg", "only_fg_noise",
    "mixed_same", "mixed_rand"
]

_image_tar_cache = {}
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


def substitute_foreground(image, bbox_list, bbox_mask_2d, patch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    h, w, c = image.shape
    bg_tiled = image.copy()

    if patch_size >= h or patch_size >= w:
        patch_size = max(4, min(h, w) // 4)

    patch = None
    for _ in range(50):
        y0 = int(rng.integers(0, h - patch_size + 1))
        x0 = int(rng.integers(0, w - patch_size + 1))
        window_mask = bbox_mask_2d[y0:y0 + patch_size, x0:x0 + patch_size]
        if not window_mask.any():
            patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
            break

    if patch is None:
        bg_pixels = image[bbox_mask_2d == 0]
        fill = bg_pixels.mean(axis=0) if len(bg_pixels) else image.reshape(-1, c).mean(axis=0)
        patch = np.tile(fill.astype(image.dtype), (patch_size, patch_size, 1))

    ph, pw = patch.shape[:2]

    for x_min, y_min, x_max, y_max in bbox_list:
        box_h, box_w = y_max - y_min, x_max - x_min
        if box_h <= 0 or box_w <= 0:
            continue
        reps_y = int(np.ceil(box_h / ph))
        reps_x = int(np.ceil(box_w / pw))
        tiled_patch = np.tile(patch, (reps_y, reps_x, 1))[:box_h, :box_w]
        bg_tiled[y_min:y_max, x_min:x_max] = tiled_patch

    return bg_tiled


def inpaint_masked_region(image, fill_mask_2d, inpaint_radius=5, method=cv2.INPAINT_TELEA):
    if not fill_mask_2d.any():
        return image.copy()
    mask_u8 = (fill_mask_2d * 255).astype(np.uint8)
    return cv2.inpaint(image, mask_u8, inpaint_radius, method)


def apply_gaussian_noise(image, mask_3d):
    noise_1d = np.random.normal(loc=128, scale=75, size=(image.shape[0], image.shape[1], 1))
    noise = np.tile(noise_1d, (1, 1, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return np.where(mask_3d == 1, noise, image)


def get_image_tar(synset):
    if synset not in _image_tar_cache:
        tar_path = os.path.join(IMAGENET_PATH, f"{synset}.tar")
        tar = tarfile.open(tar_path, "r")
        index = {m.name: m for m in tar.getmembers()}
        _image_tar_cache[synset] = (tar, index)
    return _image_tar_cache[synset]


def read_image_from_tar(synset, filename):
    tar, index = get_image_tar(synset)
    member = index.get(filename)
    if member is None:
        return None
    f = tar.extractfile(member)
    if f is None:
        return None
    data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_or_compute_mask(cls, img, bbox_list, img_stem):
    os.makedirs(os.path.join(MASK_PATH, cls), exist_ok=True)
    mask_file_path = os.path.join(MASK_PATH, cls, f"{img_stem}.png")

    if os.path.exists(mask_file_path):
        saved_mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        if saved_mask is not None:
            return (saved_mask > 127).astype(np.uint8)

    h, w = img.shape[:2]
    clamped_boxes = clamp_boxes(bbox_list, h, w)
    if not clamped_boxes:
        return None

    predictor.set_image(img)
    input_boxes = np.array(clamped_boxes)
    masks, scores, _ = predictor.predict(box=input_boxes, multimask_output=False)
    combined_mask = np.any(masks > 0, axis=0).squeeze().astype(np.uint8)

    cv2.imwrite(mask_file_path, combined_mask * 255)
    return combined_mask


def get_donor_background(cls_key, synset, img_name, max_mask_ratio):
    bg_img = read_image_from_tar(synset, img_name)
    if bg_img is None:
        return None

    bg_bboxes = get_bboxes_cached(synset)
    bg_stem = img_name.split(".")[0]
    bg_bbox_list = bg_bboxes.get(bg_stem)
    if not bg_bbox_list:
        return None

    bg_mask = get_or_compute_mask(cls_key, bg_img, bg_bbox_list, bg_stem)
    if bg_mask is not None:
        mask_ratio = np.sum(bg_mask) / bg_mask.size
        if mask_ratio <= max_mask_ratio:
            return inpaint_masked_region(bg_img, bg_mask)
        return None

    h, w = bg_img.shape[:2]
    clamped = clamp_boxes(bg_bbox_list, h, w)
    if not clamped:
        return None
    bg_bbox_mask_2d = np.zeros((h, w), dtype=np.uint8)
    for x_min, y_min, x_max, y_max in clamped:
        bg_bbox_mask_2d[y_min:y_max, x_min:x_max] = 1
    return substitute_foreground(bg_img, clamped, bg_bbox_mask_2d)


def pick_and_clean_background(cls_key, info, max_attempts=MAX_DONOR_ATTEMPTS, max_mask_ratio=0.3):
    candidates = list(info["filenames"])
    random.shuffle(candidates)

    for name in candidates[:max_attempts]:
        cleaned = get_donor_background(cls_key, info["synset"], name, max_mask_ratio)
        if cleaned is not None:
            return cleaned, name, False

    for name in candidates:
        cleaned = get_donor_background(cls_key, info["synset"], name, max_mask_ratio=1.0)
        if cleaned is not None:
            return cleaned, name, True

    return None, None, False


with open(SUPERCLASS_MAP, "r") as f:
    superclasses = json.load(f)

with open(f"{MODEL_PATH}/val_bbox_only_manifest.json", "r") as f:
    manifest_data = json.load(f)

with open(f"{MODEL_PATH}/synset_to_idx.json", "r") as f:
    synset_to_idx = json.load(f)

with open(f"{MODEL_PATH}/maps_to_idx.json", "r") as f:
    maps_to_idx = json.load(f)
idx_to_maps = {k: v for v, k in maps_to_idx.items()}

valid_filenames_by_synset = {}
for entry in manifest_data:
    if entry.get("has_bbox", False):
        filename = entry["filename"]
        synset = filename.split('_')[0]

        if synset not in valid_filenames_by_synset:
            valid_filenames_by_synset[synset] = []
        valid_filenames_by_synset[synset].append(filename)

image_paths = {}
for synset, idx in synset_to_idx.items():
    cls = idx_to_maps[int(idx)]

    filenames_for_class = sorted(valid_filenames_by_synset.get(synset, []))
    image_paths[cls] = {"synset": synset, "filenames": filenames_for_class}

    if len(filenames_for_class) == 0:
        print(f"Warning: No valid manifest images found for class {cls} (Synset: {synset})")
    elif len(filenames_for_class) < TARGET_IMAGES_PER_CLASS:
        print(f"Warning: only {len(filenames_for_class)} candidate images available for class "
              f"{cls} (target is {TARGET_IMAGES_PER_CLASS}) -- this class may fall short even "
              f"in the best case.")
    else:
        print(f"{len(filenames_for_class)} candidate images available for class {cls}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_SAM2, device=device)
predictor = SAM2ImagePredictor(sam2_model)

for variant in VARIANTS:
    os.makedirs(os.path.join(OUTPUT_PATH, variant), exist_ok=True)

quality_fallbacks = defaultdict(lambda: {"mixed_same": 0, "mixed_rand": 0})
donor_hard_failures = defaultdict(lambda: {"mixed_same": 0, "mixed_rand": 0})
images_written = defaultdict(int)

for cls, info in image_paths.items():
    candidates = info["filenames"]
    synset = info["synset"]
    class_bboxes = get_bboxes_cached(synset)

    os.makedirs(os.path.join(MASK_PATH, cls), exist_ok=True)

    successes = 0
    for img_path in candidates:
        if successes >= TARGET_IMAGES_PER_CLASS:
            break

        img_stem = img_path.split(".")[0]

        if img_stem not in class_bboxes:
            print(f"Bbox not found for {img_stem}. Trying next candidate...")
            continue

        img = read_image_from_tar(synset, img_path)
        if img is None:
            print(f"Warning: could not read {img_path} from {synset}.tar. Trying next candidate...")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox_list = clamp_boxes(class_bboxes[img_stem], img.shape[0], img.shape[1])
        if not bbox_list:
            print(f"No usable bounding box for image {img_stem}. Trying next candidate...")
            continue

        bbox_mask_2d = np.zeros(img.shape[:2], dtype=np.uint8)
        for x_min, y_min, x_max, y_max in bbox_list:
            bbox_mask_2d[y_min:y_max, x_min:x_max] = 1
        bbox_mask_3d = bbox_mask_2d[:, :, None]

        combined_mask = get_or_compute_mask(cls, img, bbox_list, img_stem)
        if combined_mask is None:
            print(f"Warning: could not obtain a mask for {img_stem}. Trying next candidate...")
            continue
        mask_3d = combined_mask[:, :, None]

        original = img.copy()

        only_bg_b = img.copy()
        only_bg_b[bbox_mask_2d == 1] = [0, 0, 0]

        only_bg_t = substitute_foreground(img, bbox_list, bbox_mask_2d)

        only_bg_noise = apply_gaussian_noise(img, bbox_mask_3d)

        no_fg = img.copy()
        no_fg[combined_mask == 1] = [0, 0, 0]

        no_fg_noise = apply_gaussian_noise(img, mask_3d)

        only_fg = img.copy()
        only_fg[combined_mask == 0] = [0, 0, 0]

        inv_mask_3d = 1 - mask_3d
        only_fg_noise = apply_gaussian_noise(img, inv_mask_3d)

        variants_dict = {
            "original": original, "only_bg_b": only_bg_b, "only_bg_t": only_bg_t,
            "only_bg_noise": only_bg_noise, "no_fg": no_fg, "no_fg_noise": no_fg_noise,
            "only_fg": only_fg, "only_fg_noise": only_fg_noise,
        }

        bg_same, same_cls_bg_name, used_fallback = pick_and_clean_background(cls, image_paths[cls])
        if bg_same is not None:
            bg_same = cv2.cvtColor(bg_same, cv2.COLOR_BGR2RGB)
            bg_same = cv2.resize(bg_same, (img.shape[1], img.shape[0]))
            variants_dict["mixed_same"] = np.where(mask_3d == 1, img, bg_same)
            if used_fallback:
                quality_fallbacks[cls]["mixed_same"] += 1
        else:
            print(f"Warning: NO usable donor at all for {cls} (mixed_same, {img_stem}) -- "
                  f"this should be extremely rare.")
            donor_hard_failures[cls]["mixed_same"] += 1

        diff_superclasses = [
            c for c in superclasses.keys()
            if superclasses[c] != superclasses[cls] and image_paths.get(c, {}).get("filenames")
        ]
        bg_rand, used_fallback_rand, rand_cls = None, False, None
        if diff_superclasses:
            rand_cls = random.choice(diff_superclasses)
            rand_info = image_paths[rand_cls]
            bg_rand, rand_cls_bg_name, used_fallback_rand = pick_and_clean_background(rand_cls, rand_info)
            if bg_rand is None:
                print(f"Warning: NO usable donor at all for {rand_cls} (mixed_rand, {img_stem}) -- "
                      f"this should be extremely rare.")
                donor_hard_failures[cls]["mixed_rand"] += 1
        else:
            print(f"Warning: no valid cross-superclass background available for {cls}. Skipping mixed_rand...")
            donor_hard_failures[cls]["mixed_rand"] += 1

        if bg_rand is not None:
            bg_rand = cv2.cvtColor(bg_rand, cv2.COLOR_BGR2RGB)
            bg_rand = cv2.resize(bg_rand, (img.shape[1], img.shape[0]))
            variants_dict["mixed_rand"] = np.where(mask_3d == 1, img, bg_rand)
            if used_fallback_rand:
                quality_fallbacks[cls]["mixed_rand"] += 1

        for v_name, v_img in variants_dict.items():
            out_folder = os.path.join(OUTPUT_PATH, v_name, cls)
            os.makedirs(out_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, img_path), cv2.cvtColor(v_img, cv2.COLOR_RGB2BGR))

        successes += 1
        images_written[cls] = successes

    if successes < TARGET_IMAGES_PER_CLASS:
        print(f"Warning: class {cls} only reached {successes}/{TARGET_IMAGES_PER_CLASS} images "
              f"-- exhausted its entire candidate pool ({len(candidates)} available). This is a "
              f"data-availability limit, not something the retry logic can fix.")

print("\n--- Summary ---")
for cls in image_paths:
    n = images_written.get(cls, 0)
    qf = quality_fallbacks[cls]
    hf = donor_hard_failures[cls]
    if n < TARGET_IMAGES_PER_CLASS or qf["mixed_same"] or qf["mixed_rand"] or hf["mixed_same"] or hf["mixed_rand"]:
        print(f"  {cls}: {n}/{TARGET_IMAGES_PER_CLASS} written | "
              f"quality-fallback used -- mixed_same: {qf['mixed_same']}, mixed_rand: {qf['mixed_rand']} | "
              f"hard failures -- mixed_same: {hf['mixed_same']}, mixed_rand: {hf['mixed_rand']}")