import os
import io
import json
import glob
import random
import zipfile

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms

VERSION = "v4"
MODEL_RATIOS = [0, 25, 50, 75, 100]
MODEL_NAMES = ["resnet", "convnext"]

BASE_DIR = "/work/dldevel/trieba/bachelor_thesis"
DATASET_PATH = f"{BASE_DIR}/training_data/eval_dataset"
MAPS_IMAGE_ROOT = "/work/dldevel/galella/datasets/MAPS/dataset_5k/images"
MAPS_HOLDOUT_MANIFEST_PATH = f"{BASE_DIR}/training_data/misc/maps_holdout_manifest.json"

OUTPUT_DIR = f"{BASE_DIR}/results/evaluation/pca_{VERSION}"

LAYER_KEYS = ["layer1", "layer2", "layer3", "layer4"]
N_CLASSES = 5
N_IMAGES_PER_CLASS_DOMAIN = 30
CLASS_SELECTION_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_name, ratio):
    checkpoint_path = f"{BASE_DIR}/results/model_{ratio}/{model_name}/{VERSION}/best_checkpoint.pth"
    with open(f"{BASE_DIR}/results/model_{ratio}/{model_name}/{VERSION}/maps_to_idx.json", "r") as f:
        maps_to_idx = json.load(f)
    num_classes = len(maps_to_idx)

    if model_name == "resnet":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        layer_modules = {
            "layer1": model.layer1[-1], "layer2": model.layer2[-1],
            "layer3": model.layer3[-1], "layer4": model.layer4[-1],
        }
    elif model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        layer_modules = {
            "layer1": model.features[1][-1], "layer2": model.features[3][-1],
            "layer3": model.features[5][-1], "layer4": model.features[7][-1],
        }
    else:
        raise ValueError(model_name)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model, layer_modules


def load_canonical_class_names():
    path = f"{BASE_DIR}/results/model_{MODEL_RATIOS[0]}/{MODEL_NAMES[0]}/{VERSION}/maps_to_idx.json"
    with open(path, "r") as f:
        return sorted(json.load(f).keys())


class FeatureExtractor:
    def __init__(self, layer_modules):
        self.activations = {}
        self.handles = [
            module.register_forward_hook(self._make_hook(key))
            for key, module in layer_modules.items()
        ]

    def _make_hook(self, key):
        def hook(module, inp, out):
            pooled = out.mean(dim=(2, 3)) if out.dim() == 4 else out.mean(dim=1)
            self.activations[key] = pooled.detach().cpu().numpy()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


_zip_cache = {}


def _get_zip(path):
    if path not in _zip_cache:
        _zip_cache[path] = zipfile.ZipFile(path, "r")
    return _zip_cache[path]


def build_image_refs(class_names, manifest, rng):
    """{(cls_name, domain): [ref, ...]} for domain in {"maps", "imagenet_original"}."""
    refs = {}
    for cls_name in class_names:
        members = manifest.get(cls_name, [])
        if len(members) > N_IMAGES_PER_CLASS_DOMAIN:
            members = rng.sample(members, N_IMAGES_PER_CLASS_DOMAIN)
        zip_path = os.path.join(MAPS_IMAGE_ROOT, cls_name, f"{cls_name}_001.zip")
        refs[(cls_name, "maps")] = [("maps", zip_path, m) for m in members]

        paths = sorted(glob.glob(os.path.join(DATASET_PATH, "original", cls_name, "*")))
        if len(paths) > N_IMAGES_PER_CLASS_DOMAIN:
            paths = rng.sample(paths, N_IMAGES_PER_CLASS_DOMAIN)
        refs[(cls_name, "imagenet_original")] = [("file", p, None) for p in paths]
    return refs


def load_image(ref):
    kind, a, b = ref
    if kind == "maps":
        img_bytes = _get_zip(a).read(b)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return Image.open(a).convert("RGB")


def extract_per_image_features(model, extractor, refs, batch_size=32):
    """Returns {layer: {(cls_name, domain): (N, C) array}}."""
    out = {layer: {} for layer in LAYER_KEYS}
    for (cls_name, domain), img_refs in refs.items():
        collected = {layer: [] for layer in LAYER_KEYS}
        for start in range(0, len(img_refs), batch_size):
            batch_refs = img_refs[start:start + batch_size]
            imgs = torch.stack([transform(load_image(r)) for r in batch_refs]).to(device)
            with torch.no_grad():
                model(imgs)
            for layer in LAYER_KEYS:
                collected[layer].append(extractor.activations[layer])
        for layer in LAYER_KEYS:
            out[layer][(cls_name, domain)] = np.concatenate(collected[layer], axis=0) if collected[layer] else \
                np.empty((0, 1))
    return out


def pca_2d(X):
    return PCA(n_components=2).fit_transform(X)


def lighten(color, amount=0.55):
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (np.array([1, 1, 1]) - c) * amount)


def darken(color, amount=0.35):
    c = np.array(mcolors.to_rgb(color))
    return tuple(c * (1 - amount))


def plot_model_ratio(features_by_layer, class_names, title, out_path):
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(class_names)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    handles, labels = None, None

    for ax, layer in zip(axes, LAYER_KEYS):
        layer_data = features_by_layer[layer]
        keys = list(layer_data.keys())
        sizes = [layer_data[k].shape[0] for k in keys]
        stacked = np.concatenate([layer_data[k] for k in keys], axis=0)
        coords_2d = pca_2d(stacked)

        cur_handles, cur_labels = [], []
        offset = 0
        for (cls_name, domain), n in zip(keys, sizes):
            pts = coords_2d[offset:offset + n]
            offset += n
            cls_idx = class_names.index(cls_name)
            base = base_colors[cls_idx]
            if domain == "maps":
                color, marker, dom_label = darken(base), "^", "MAPS"
            else:
                color, marker, dom_label = lighten(base), "o", "ImageNet"
            h = ax.scatter(pts[:, 0], pts[:, 1], color=[color], marker=marker, s=28,
                            edgecolors="black", linewidths=0.3, alpha=0.85)
            cur_handles.append(h)
            cur_labels.append(f"{cls_name} ({dom_label})")

        ax.set_title(layer, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if handles is None:
            handles, labels = cur_handles, cur_labels

    fig.suptitle(title, fontsize=13)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    with open(MAPS_HOLDOUT_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    canonical_class_names = load_canonical_class_names()
    rng = random.Random(CLASS_SELECTION_SEED)
    selected_classes = sorted(rng.sample(canonical_class_names, N_CLASSES))
    print(f"Selected classes: {selected_classes}")

    os.makedirs(os.path.join(OUTPUT_DIR, "pca"), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "pca", "selected_classes.json"), "w") as f:
        json.dump(selected_classes, f, indent=2)

    refs = build_image_refs(selected_classes, manifest, rng)

    for model_name in MODEL_NAMES:
        for ratio in MODEL_RATIOS:
            print(f"\n=== {model_name} @ {ratio}% MAPS ===")
            try:
                model, layer_modules = load_model(model_name, ratio)
            except Exception as e:
                print(f"  skipping ({e})")
                continue

            extractor = FeatureExtractor(layer_modules)
            features_by_layer = extract_per_image_features(model, extractor, refs)
            extractor.remove()

            out_path = os.path.join(OUTPUT_DIR, "pca", model_name, f"{ratio}.png")
            plot_model_ratio(features_by_layer, selected_classes,
                              title=f"{model_name}, {ratio}% MAPS", out_path=out_path)
            print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()