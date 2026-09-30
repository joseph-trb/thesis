import os
import io
import csv
import sys
import json
import time
import torch
import zipfile
import tarfile
import torch.nn as nn
import multiprocessing as mp
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from PIL import Image
from torch import optim
from collections import defaultdict
from torchvision import models
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, Dataset

def train_and_val(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, results_path):

    model.to(device)
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []
    }
    len_train = len(train_loader.dataset)
    len_val = len(val_loader.dataset)
    num_train_batches = len(train_loader)
    best_acc = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

            if batch_idx % 50 == 0 or batch_idx == num_train_batches:
                current_batch_loss = loss.item()
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Step [{batch_idx:04d}/{num_train_batches:04d}] | "
                    f"Batch Loss: {current_batch_loss:.4f}",
                    flush=True
                )

        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = running_loss / len_train
        epoch_train_acc = running_corrects / len_train

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()

        epoch_val_loss = val_loss / len_val
        epoch_val_acc = val_corrects / len_val
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        torch.save(model.state_dict(), f"{results_path}/last_checkpoint.pth")

        saved_best_str = ""
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f"{results_path}/best_checkpoint.pth")
            saved_best_str = f" [New Best Checkpoint: {best_acc * 100:.2f}%]"

        epoch_time = time.time() - epoch_start_time
        print(
            f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] Summary (Time: {epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}{saved_best_str}\n",
            flush=True
        )

    total_time_min = (time.time() - start_time) / 60
    print(f"Training completed in {total_time_min:.2f} mins. Peak Real Val Acc: {best_acc * 100:.2f}%\n", flush=True)

    return history

def plot_training_curves(history, results_path):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=1.8)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='tab:red', linestyle='--', linewidth=1.8)
    ax1.set_title("Cross-Entropy Loss per Epoch", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc', color='tab:blue', linewidth=1.8)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Acc', color='tab:green', linestyle='--', linewidth=1.8)
    ax2.set_title("Top-1 Accuracy per Epoch (%)", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/training_metrics.png", dpi=300)
    plt.close()
    print(f"Metrics plot saved!")

def create_class_mappings(labels_csv_path, class_index_json_path, selected_classes):
    with open(class_index_json_path, "r") as f:
        in_index_data = json.load(f)  # {"402": ["n02672880", "acoustic_guitar"]}
    
    idx_to_synset = {int(k): v[0] for k, v in in_index_data.items()}

    maps_name_to_in_idx = {}
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            if not row:
                continue
            maps_name_to_in_idx[row[0].strip()] = int(row[1].strip())

    maps_to_synset = {}
    for name in selected_classes:
        in_idx = maps_name_to_in_idx[name]
        synset_id = idx_to_synset[in_idx]
        maps_to_synset[name] = synset_id

    unique_synsets = sorted(list(maps_to_synset.values()))
    synset_to_target_idx = {synset: idx for idx, synset in enumerate(unique_synsets)}
    
    maps_name_to_target_idx = {name: synset_to_target_idx[synset] for name, synset in maps_to_synset.items()}

    return synset_to_target_idx, maps_name_to_target_idx

class MAPSDataset(Dataset):

    def __init__(self, root_dir, maps_to_target_idx, transform=None):
        self.samples = []  # (zip_path, member_filename, target_idx)
        self.transform = transform
        self._zip_cache = {}

        for class_name, target_idx in maps_to_target_idx.items():
            path = os.path.join(root_dir, class_name, f"{class_name}_001.zip")
            if not os.path.isfile(path):
                print(f"Warning: Zip not found: {path}")
                continue

            with zipfile.ZipFile(path, 'r') as zf:
                for member_name in zf.namelist():
                    if member_name.lower().endswith(".png"):
                        self.samples.append((path, member_name, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, member_name, target = self.samples[idx]

        if path not in self._zip_cache:
            self._zip_cache[path] = zipfile.ZipFile(path, 'r')

        zf = self._zip_cache[path]
        img_bytes = zf.read(member_name)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

class ImageNetDataset(Dataset):

    def __init__(self, root_dir, synset_to_target_idx, transform=None):
        self.samples = []
        self.transform = transform
        self._tar_cache = {}

        for synset, target_idx in synset_to_target_idx.items():
            path = os.path.join(root_dir, f"{synset}.tar")
            if not os.path.exists(path):
                print(f"Warning: Tar not found: {path}")
                continue

            with tarfile.open(path, "r:") as archive:
                for member in archive.getmembers():
                    if member.name.lower().endswith('.jpeg'):
                        self.samples.append((path, member.name, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, member_name, target = self.samples[idx]
        if path not in self._tar_cache:
            self._tar_cache[path] = tarfile.open(path, "r:")

        archive = self._tar_cache[path]
        file_obj = archive.extractfile(member_name)
        img = Image.open(io.BytesIO(file_obj.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

class TransformedDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, target = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

def get_balanced_class_subset(dataset, samples_per_class, generator=None):
    class_to_indices = defaultdict(list)

    if isinstance(dataset, Subset):
        for subset_idx, orig_idx in enumerate(dataset.indices):
            target = dataset.dataset.samples[orig_idx][-1]
            class_to_indices[target].append(subset_idx)
    elif hasattr(dataset, "samples"):
        for idx, item in enumerate(dataset.samples):
            class_to_indices[item[-1]].append(idx)
    else:
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            class_to_indices[target].append(idx)

    selected_indices = []
    for cls_idx, indices in class_to_indices.items():
        indices_tensor = torch.tensor(indices)
        perm = indices_tensor[torch.randperm(len(indices_tensor), generator=generator)]
        
        take_n = min(samples_per_class, len(perm))
        if take_n < samples_per_class:
            print(f"Warning: Class {cls_idx} only has {len(perm)} images (requested {samples_per_class})")
            
        selected_indices.extend(perm[:take_n].tolist())

    return Subset(dataset, selected_indices)

def split_imagenet(raw_dataset, bbox_filenames, save_dir, n_train=1000, n_val=300, generator=None):
    os.makedirs(save_dir, exist_ok=True)
    class_to_items = defaultdict(list)

    for idx, item in enumerate(raw_dataset.samples):
        # (tar_path, member_name, target_idx) or (file_path, target_idx)
        member_name = item[1] if len(item) == 3 else item[0]
        filename = os.path.basename(member_name)
        target = item[-1]

        has_bbox = filename in bbox_filenames
        class_to_items[target].append({
            'dataset_idx': idx,
            'filename': filename,
            'member_name': member_name,
            'has_bbox': has_bbox,
            'class_idx': target
        })

    train_indices = []
    val_indices = []
    val_metadata = []

    for cls_idx, items in class_to_items.items():
        with_bbox = [it for it in items if it['has_bbox']]
        without_bbox = [it for it in items if not it['has_bbox']]

        if generator is not None:
            perm_bbox = torch.randperm(len(with_bbox), generator=generator).tolist()
            with_bbox = [with_bbox[i] for i in perm_bbox]

            perm_nobbox = torch.randperm(len(without_bbox), generator=generator).tolist()
            without_bbox = [without_bbox[i] for i in perm_nobbox]

        val_pool = with_bbox + without_bbox
        selected_val = val_pool[:n_val]
        remaining_train = val_pool[n_val:n_val + n_train]

        val_indices.extend([it['dataset_idx'] for it in selected_val])
        train_indices.extend([it['dataset_idx'] for it in remaining_train])
        val_metadata.extend(selected_val)

    val_ids_export = [
        {
            "filename": it['filename'],
            "class_idx": it['class_idx'],
            "has_bbox": it['has_bbox'],
            "source_path": it['member_name']
        }
        for it in val_metadata
    ]

    with open(os.path.join(save_dir, "val_image_manifest.json"), "w") as f:
        json.dump(val_ids_export, f, indent=4)

    bbox_only_manifest = [it for it in val_ids_export if it["has_bbox"]]
    with open(os.path.join(save_dir, "val_bbox_only_manifest.json"), "w") as f:
        json.dump(bbox_only_manifest, f, indent=4)

    print(f"Validation dataset: {len(val_indices)} total images saved.")
    print(f"Bounding box evaluation samples: {len(bbox_only_manifest)} images available.")

    return Subset(raw_dataset, train_indices), Subset(raw_dataset, val_indices)

def load_bbox_filenames(annotations_root, target_synsets):
    bbox_filenames = set()
    
    for synset in target_synsets:
        path = os.path.join(annotations_root, f"{synset}.tar.gz")
        if not os.path.isfile(path):
            print(f"Warning: Annotation archive not found for synset {synset}")
            continue

        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                if member.name.lower().endswith(".xml"):
                    base_name = os.path.splitext(os.path.basename(member.name))[0]
                    bbox_filenames.add(f"{base_name}.JPEG")

    return bbox_filenames


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available!")

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(SEED)

    # adjust RATIO_MAPS [0, 1] and MODEL_RATIO [0, 100] to train different model

    TOTAL_SAMPLES_PER_CLASS = 1000
    RATIO_MAPS = 0.25
    MODEL_RATIO = 25                # just for naming of the folder
    MODEL_NAME = "resnet"
    VERSION = "v4"

    RESULTS_DIR = f"/work/dldevel/trieba/bachelor_thesis/results/model_{MODEL_RATIO}/{MODEL_NAME}/{VERSION}"

    n_maps = int(TOTAL_SAMPLES_PER_CLASS * RATIO_MAPS)
    n_imagenet = TOTAL_SAMPLES_PER_CLASS - n_maps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    os.makedirs(f'{RESULTS_DIR}', exist_ok=True)

    MEAN_FILL = (124, 116, 104)     # ImageNet mean * 255

    transform_train_in = T.Compose([
        T.RandomZoomOut(fill=MEAN_FILL, side_range=(1.0, 1.35), p=0.5),
        T.Resize((224, 224), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_train_maps = T.Compose([
        T.RandomApply([T.JPEG(quality=(30, 95))], p=0.5),
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(1.0, 1.0), antialias=True),
        T.RandomZoomOut(fill=MEAN_FILL, side_range=(1.0, 1.35), p=0.5),
        T.Resize((224, 224), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.3),
        T.AugMix(severity=3, mixture_width=3, all_ops=False),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    LABELS_CSV = "/work/dldevel/galella/datasets/MAPS/dataset_5k/labels.csv"
    IMAGENET_JSON = "imagenet_class_index.json"
    MAPS_DIR = "/work/dldevel/galella/datasets/MAPS/dataset_5k/images"
    IMAGENET_DIR = "/work/dldevel/galella/datasets/ImageNet/ILSVRC2012_img_train"
    ANNOTATIONS_ROOT = "/work/dldevel/trieba/bachelor_thesis/training_data/ImageNet_BoundingBoxes/Annotations"

    SELECTED_CLASSES= [
        "ambulance", "jeep", "limousine", "minibus", "mountain-bike", "taxi",
        "acoustic-guitar", "banjo", "drum", "electric-guitar", "grand-piano",
        "harp", "ocarina", "pipe-organ", "violoncello",
        "ant", "bee", "dragonfly", "mantis",
        "clownfish", "goldfish", "stingray", "white-shark",
        "alligator", "komodo-dragon",
        "chihuahua", "cheetah", "robin",
        "broccoli", "cucumber", "mushroom", "pineapple",
        "pizza", "strawberry", "obelisk", "daisy", 
        "abacus", "backpack", "balloon", "basketball", "binoculars",
        "coffee-mug", "computer-keyboard", "computer-mouse", "dumbbell",
        "hourglass", "joystick", "microphone", "screwdriver", "volleyball"
    ]

    synset_to_idx, maps_to_idx = create_class_mappings(LABELS_CSV, IMAGENET_JSON, SELECTED_CLASSES)

    with open("class_to_idx.json", "w") as f:
        json.dump(synset_to_idx, f, indent=4)

    raw_maps_dataset = MAPSDataset(
        root_dir=MAPS_DIR,
        maps_to_target_idx=maps_to_idx
    )
    raw_imagenet_dataset = ImageNetDataset(
        root_dir=IMAGENET_DIR,
        synset_to_target_idx=synset_to_idx
    )


    bbox_filenames = load_bbox_filenames(
        annotations_root=ANNOTATIONS_ROOT,
        target_synsets=list(synset_to_idx.keys())
    )


    imagenet_train_pool, imagenet_val = split_imagenet(
        raw_dataset=raw_imagenet_dataset,
        bbox_filenames=bbox_filenames,
        save_dir=RESULTS_DIR,
        n_train=1000,
        n_val=300,
        generator=generator
    )

    imagenet_train = get_balanced_class_subset(
        imagenet_train_pool, 
        n_imagenet, 
        generator=generator
    )
    print("ImageNet training images:", len(imagenet_train))
    maps_train = get_balanced_class_subset(
        raw_maps_dataset, 
        n_maps, 
        generator=generator
    )
    print("MAPS training images:", len(maps_train))

    train_data_in = TransformedDataset(imagenet_train, transform=transform_train_in)
    train_data_maps = TransformedDataset(maps_train, transform=transform_train_maps)
    train_data = ConcatDataset([train_data_in, train_data_maps])
    val_data = TransformedDataset(imagenet_val, transform=transform_val)
    print("training data:", len(train_data))
    print("validation data:", len(val_data))


    BATCH_SIZE = 256
    EPOCHS = 90

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    len_train = len(train_data)
    len_val = len(val_data)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS - warmup_epochs, 
        eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )


    with open(f"{RESULTS_DIR}/config.json", "w") as f:
        config = {
            "model": MODEL_NAME,
            "MAPS_ratio": RATIO_MAPS,
            "training_size": len_train,
            "training_ImageNet": len(imagenet_train),
            "training_MAPS": len(maps_train),
            "validation_size": len_val,
            "batch_size": BATCH_SIZE,
            "loss_function": "Cross-Entropy",
            "optimizer": "sgd",
            "momentum": 0.9,
            "learning_rate": 0.1,
            "weight_decay": 1e-4,
            "applied_augmentations": [
                "RandomZoomOut", "Resize", "RandomHorizontalFlip", "ColorJitter", "GaussianBlur"
            ],
            "classes": SELECTED_CLASSES
        }
        json.dump(config, f, indent=4)


    with open(f"{RESULTS_DIR}/synset_to_idx.json", "w") as f:
        json.dump(synset_to_idx, f, indent=4)

    with open(f"{RESULTS_DIR}/maps_to_idx.json", "w") as f:
        json.dump(maps_to_idx, f, indent=4)
        

    print("starting training!")
    history = train_and_val(EPOCHS, model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, RESULTS_DIR)

    with open(f"{RESULTS_DIR}/history.json", "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(history, RESULTS_DIR)

if __name__ == '__main__':
    mp.set_start_method("fork", force=True)
    n_cpu = len(os.sched_getaffinity(0))
    print("cpus visible:", n_cpu)
    main()