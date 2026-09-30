import os
import csv
import cv2
import json
import torch
import tarfile
import numpy as np
import multiprocessing as mp
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

VERSION = "v1"
MODEL_VERSION = "v4"
MODEL_RATIOS = [0, 12, 25, 37, 50, 62, 75, 80, 85, 90, 95, 99, 100]
MODEL_NAMES = ["resnet", "convnext"]
EVAL_DATASET_VARIANTS = ["original", "mixed_same", "mixed_rand", "only_fg", "only_fg_noise"]
IMAGE_COUNT_TARGET = 300

BASE_DIR = "/work/dldevel/trieba/bachelor_thesis"
DATASET_PATH = f"{BASE_DIR}/training_data/eval_dataset"
MASK_PATH = f"{BASE_DIR}/training_data/SAM/ImageNet_validation_masks"
BBOX_PATH = f"{BASE_DIR}/training_data/ImageNet_BoundingBoxes/Annotations"
IMAGENET_PATH = "/work/dldevel/galella/datasets/ImageNet/ILSVRC2012_img_train"
OUTPUT_PATH = f"{BASE_DIR}/results/evaluation/hirescam_metrics_{VERSION}"

CHECKPOINT_SAM2 = f"{BASE_DIR}/training_data/SAM/checkpoint/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], original_tuple[1], path


class HeldOutDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples  # list of (cls_name, synset, img_name)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_name, synset, img_name = self.samples[idx]
        img_bgr = read_image_from_tar(synset, img_name)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_stem = img_name.split(".")[0]
        return self.transform(Image.fromarray(img)), cls_name, img_stem



_bbox_cache = {}
_image_tar_cache = {}


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


def get_or_compute_mask(cls_name, synset, img_stem, predictor):
    os.makedirs(os.path.join(MASK_PATH, cls_name), exist_ok=True)
    mask_file = os.path.join(MASK_PATH, cls_name, f"{img_stem}.png")

    if os.path.exists(mask_file):
        saved = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if saved is not None:
            return (saved > 127).astype(np.uint8)

        print(f"[MASK READ FAILED] {mask_file}")

    bboxes = get_bboxes_cached(synset).get(img_stem)
    if not bboxes:
        print(
            f"[NO BBOX MATCH] "
            f"class={cls_name}, synset={synset}, img_stem={img_stem}"
        )
        return None

    img_bgr = read_image_from_tar(synset, f"{img_stem}.JPEG")
    if img_bgr is None:
        print(
            f"[IMAGE READ FAILED] "
            f"synset={synset}, filename={img_stem}"
        )
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    clamped = clamp_boxes(bboxes, img.shape[0], img.shape[1])
    if not clamped:
        print(
            f"[INVALID BBOX] "
            f"class={cls_name}, synset={synset}, "
            f"img={img_stem}, image_shape={img.shape}, "
            f"original_boxes={bboxes}"
        )
        return None

    predictor.set_image(img)
    masks, _, _ = predictor.predict(box=np.array(clamped), multimask_output=False)
    combined_mask = np.any(masks > 0, axis=0).squeeze().astype(np.uint8)

    cv2.imwrite(mask_file, combined_mask * 255)
    return combined_mask

def build_evaluation_sample_list():
    model_path = f"{BASE_DIR}/results/model_0/resnet/{MODEL_VERSION}"

    with open(f"{model_path}/val_bbox_only_manifest.json", "r") as f:
        manifest_data = json.load(f)
    with open(f"{model_path}/synset_to_idx.json", "r") as f:
        synset_to_idx = json.load(f)
    with open(f"{model_path}/maps_to_idx.json", "r") as f:
        maps_to_idx = json.load(f)
    idx_to_maps = {k: v for v, k in maps_to_idx.items()}

    valid_filenames_by_synset = defaultdict(list)
    for entry in manifest_data:
        if entry.get("has_bbox", False):
            filename = entry["filename"]
            synset = filename.split('_')[0]
            valid_filenames_by_synset[synset].append(filename)

    samples = []  # (cls_name, synset, img_name)
    for synset, idx in synset_to_idx.items():
        cls_name = idx_to_maps[int(idx)]
        filenames = sorted(valid_filenames_by_synset.get(synset, []))[:IMAGE_COUNT_TARGET]
        for img_name in filenames:
            samples.append((cls_name, synset, img_name))

    return samples


def complete_evaluation_masks(samples, predictor):
    computed, already_cached, failed = 0, 0, 0

    for cls_name, synset, img_name in samples:
        img_stem = img_name.split(".")[0]
        mask_file = os.path.join(MASK_PATH, cls_name, f"{img_stem}.png")
        if os.path.exists(mask_file):
            already_cached += 1
            continue

        mask = get_or_compute_mask(cls_name, synset, img_stem, predictor)
        if mask is None:
            print(f"Warning: could not segment {cls_name}/{img_name} for the held-out pool.")
            failed += 1
        else:
            computed += 1

    print(f"\nHeld-out mask completion: {already_cached} already cached, "
          f"{computed} newly computed, {failed} failed.\n")


def load_model(model_name, ratio):
    checkpoint_path = f"{BASE_DIR}/results/model_{ratio}/{model_name}/{MODEL_VERSION}/best_checkpoint.pth"
    with open(f"{BASE_DIR}/results/model_{ratio}/{model_name}/{MODEL_VERSION}/maps_to_idx.json", "r") as f:
        maps_to_idx = json.load(f)
    idx_to_maps = {int(v): k for k, v in maps_to_idx.items()}
    num_classes = len(maps_to_idx)

    if model_name == "resnet":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        target_layers = [model.features[7][-1]]

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, target_layers, maps_to_idx, idx_to_maps


def compute_metrics_batched(cam_batch, mask_batch):
    cam_pos = np.maximum(cam_batch, 0)
    sum_cam = cam_pos.sum(axis=(1, 2))
    ebpg_scores, auc_scores = [], []

    for i in range(len(cam_batch)):
        if sum_cam[i] > 0:
            ebpg = (cam_pos[i] * mask_batch[i]).sum() / sum_cam[i]
        else:
            ebpg = 0.0
        ebpg_scores.append(float(ebpg))

        y_true = mask_batch[i].flatten()
        y_score = cam_batch[i].flatten()

        if np.unique(y_true).size < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_score)

        auc_scores.append(float(auc))

    return ebpg_scores, auc_scores


def evaluate_batch(images, cls_names, img_stems, model, cam_engine, maps_to_idx,
                    idx_to_maps, predictor, record_sink, class_metrics, skip_counts,
                    source_label, variant_label, model_name, ratio, track_skips):
    images = images.to(device, non_blocking=True)
    batch_size = images.size(0)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)

    masks_list, valid_idx, true_targets = [], [], []
    for i in range(batch_size):
        cls_name = cls_names[i]
        img_stem = img_stems[i]
        synset = img_stem.split('_')[0]

        mask = get_or_compute_mask(cls_name, synset, img_stem, predictor)
        if mask is None:
            if track_skips:
                skip_counts[(source_label, variant_label, cls_name)] += 1
            continue

        resized_mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        masks_list.append(resized_mask)
        valid_idx.append(i)
        true_targets.append(ClassifierOutputTarget(int(maps_to_idx[cls_name])))

    if not valid_idx:
        return

    valid_images = images[valid_idx]
    mask_batch = np.stack(masks_list)

    true_cam = cam_engine(input_tensor=valid_images, targets=true_targets)
    true_ebpg, true_auc = compute_metrics_batched(true_cam, mask_batch)

    pred_idx_list = [int(predicted[i].item()) for i in valid_idx]
    true_idx_list = [int(maps_to_idx[cls_names[i]]) for i in valid_idx]
    mis_local = [k for k, (p, t) in enumerate(zip(pred_idx_list, true_idx_list)) if p != t]

    pred_ebpg, pred_auc = list(true_ebpg), list(true_auc)
    if mis_local:
        mis_images = valid_images[mis_local]
        mis_masks = mask_batch[mis_local]
        mis_targets = [ClassifierOutputTarget(pred_idx_list[k]) for k in mis_local]
        mis_cam = cam_engine(input_tensor=mis_images, targets=mis_targets)
        mis_ebpg, mis_auc = compute_metrics_batched(mis_cam, mis_masks)
        for local_k, k in enumerate(mis_local):
            pred_ebpg[k] = mis_ebpg[local_k]
            pred_auc[k] = mis_auc[local_k]

    for k, i in enumerate(valid_idx):
        cls_name = cls_names[i]
        true_idx = true_idx_list[k]
        pred_idx = pred_idx_list[k]
        is_correct = pred_idx == true_idx

        class_metrics[cls_name]["true_ebpg"].append(true_ebpg[k])
        class_metrics[cls_name]["true_auc"].append(true_auc[k])
        class_metrics[cls_name]["pred_ebpg"].append(pred_ebpg[k])
        class_metrics[cls_name]["pred_auc"].append(pred_auc[k])

        record_sink.append({
            "model_name": model_name,
            "ratio": ratio,
            "source": source_label,
            "variant": variant_label,
            "img_stem": img_stems[i],
            "true_class": cls_name,
            "predicted_class": idx_to_maps.get(pred_idx, str(pred_idx)),
            "correct": is_correct,
            "true_class_prob": float(probs[i, true_idx].item()),
            "predicted_class_prob": float(probs[i, pred_idx].item()),
            "true_target_ebpg": true_ebpg[k],
            "true_target_auc": true_auc[k],
            "predicted_target_ebpg": pred_ebpg[k],
            "predicted_target_auc": pred_auc[k],
        })


def summarize_class_metrics(class_metrics):
    per_class = {}
    all_true_ebpg, all_true_auc, all_pred_ebpg, all_pred_auc = [], [], [], []
    for cls_name, vals in class_metrics.items():
        per_class[cls_name] = {
            "True_EBPG": float(np.mean(vals["true_ebpg"])),
            "True_AUC": float(np.nanmean(vals["true_auc"])),
            "Predicted_EBPG": float(np.mean(vals["pred_ebpg"])),
            "Predicted_AUC": float(np.nanmean(vals["pred_auc"])),
            "N": len(vals["true_ebpg"]),
        }
        all_true_ebpg.extend(vals["true_ebpg"])
        all_true_auc.extend(vals["true_auc"])
        all_pred_ebpg.extend(vals["pred_ebpg"])
        all_pred_auc.extend(vals["pred_auc"])

    sample_weighted = {
        "True_EBPG": float(np.mean(all_true_ebpg)) if all_true_ebpg else 0.0,
        "True_AUC": float(np.nanmean(all_true_auc)) if all_true_auc else 0.0,
        "Predicted_EBPG": float(np.mean(all_pred_ebpg)) if all_pred_ebpg else 0.0,
        "Predicted_AUC": float(np.nanmean(all_pred_auc)) if all_pred_auc else 0.0,
    }
    class_weighted = {
        "True_EBPG": float(np.mean([c["True_EBPG"] for c in per_class.values()])) if per_class else 0.0,
        "True_AUC": float(np.nanmean([c["True_AUC"] for c in per_class.values()])) if per_class else 0.0,
        "Predicted_EBPG": float(np.mean([c["Predicted_EBPG"] for c in per_class.values()])) if per_class else 0.0,
        "Predicted_AUC": float(np.nanmean([c["Predicted_AUC"] for c in per_class.values()])) if per_class else 0.0,
    }
    return per_class, sample_weighted, class_weighted


def write_csv(path, records):
    if not records:
        print(f"Note: no records to write for {path}")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_SAM2, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    evaluation_samples = build_evaluation_sample_list()
    complete_evaluation_masks(evaluation_samples, predictor)

    _image_tar_cache.clear()

    evaluation_dataset = HeldOutDataset(evaluation_samples, transform)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    results = defaultdict(dict)
    all_records = []
    skip_counts = defaultdict(int)

    for model_name in MODEL_NAMES:
        for ratio in MODEL_RATIOS:
            print(f"\nEvaluating HiResCAM on {model_name} (Ratio: {ratio})")
            try:
                model, target_layers, maps_to_idx, idx_to_maps = load_model(model_name, ratio)
            except Exception as e:
                print(f"Skipping {model_name}_{ratio}: {e}")
                continue

            cam_engine = HiResCAM(model=model, target_layers=target_layers)
            model_key = f"{model_name}_{ratio}"
            track_skips = (model_name == MODEL_NAMES[0] and ratio == MODEL_RATIOS[0])

            for variant in EVAL_DATASET_VARIANTS:
                variant_dir = os.path.join(DATASET_PATH, variant)
                if not os.path.exists(variant_dir):
                    continue

                dataset = ImageFolderWithPaths(root=variant_dir, transform=transform)
                dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
                class_metrics = defaultdict(lambda: {"true_ebpg": [], "true_auc": [], "pred_ebpg": [], "pred_auc": []})

                for images, folder_indices, paths in dataloader:
                    cls_names = [dataset.classes[int(fi)] for fi in folder_indices]
                    img_stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
                    evaluate_batch(images, cls_names, img_stems, model, cam_engine,
                                    maps_to_idx, idx_to_maps, predictor, all_records,
                                    class_metrics, skip_counts, "eval_dataset", variant,
                                    model_name, ratio, track_skips)

                per_class, sample_weighted, class_weighted = summarize_class_metrics(class_metrics)
                results[model_key][f"eval_dataset/{variant}"] = {
                    "SampleWeighted": sample_weighted,
                    "ClassWeighted": class_weighted,
                    "PerClass": per_class,
                }
                print(f"  eval_dataset/{variant.ljust(12)} -> "
                      f"True_EBPG(cb): {class_weighted['True_EBPG']:.3f} | "
                      f"True_AUC(cb): {class_weighted['True_AUC']:.3f} | "
                      f"Pred_EBPG(cb): {class_weighted['Predicted_EBPG']:.3f}")

            class_metrics = defaultdict(lambda: {"true_ebpg": [], "true_auc": [], "pred_ebpg": [], "pred_auc": []})
            for images, cls_names, img_stems in evaluation_loader:
                evaluate_batch(images, list(cls_names), list(img_stems), model, cam_engine,
                                maps_to_idx, idx_to_maps, predictor, all_records,
                                class_metrics, skip_counts, "evaluation", "original",
                                model_name, ratio, track_skips)

            per_class, sample_weighted, class_weighted = summarize_class_metrics(class_metrics)
            results[model_key]["evaluation/original"] = {
                "SampleWeighted": sample_weighted,
                "ClassWeighted": class_weighted,
                "PerClass": per_class,
            }
            print(f"  evaluation/original      -> "
                  f"True_EBPG(cb): {class_weighted['True_EBPG']:.3f} | "
                  f"True_AUC(cb): {class_weighted['True_AUC']:.3f}")

    with open(f"{OUTPUT_PATH}/hirescam_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    write_csv(f"{OUTPUT_PATH}/per_sample_records.csv", all_records)

    misclassified = [r for r in all_records if not r["correct"]]
    write_csv(f"{OUTPUT_PATH}/misclassifications.csv", misclassified)

    print("\n--- Skipped (no mask available) counts, first model only ---")
    for (source, variant, cls_name), n in sorted(skip_counts.items()):
        print(f"  {source}/{variant}/{cls_name}: {n} skipped")

    print(f"\nSaved CAM metrics to {OUTPUT_PATH}/hirescam_eval_results.json")
    print(f"Saved {len(all_records)} per-sample records to {OUTPUT_PATH}/per_sample_records.csv")
    print(f"Saved {len(misclassified)} misclassification records to {OUTPUT_PATH}/misclassifications.csv")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()