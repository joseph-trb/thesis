import os
import json
import torch
import random
import multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    MODEL_RATIOS = [0, 12, 25, 37, 50, 62, 75, 80, 85, 90, 95, 99, 100]
    for MODEL_RATIO in MODEL_RATIOS:
        SEED = 0
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        VERSION = "v4"

        VERSION_RESNET = "v4"
        MODEL_RESNET_PATH = f"/work/dldevel/trieba/bachelor_thesis/results/model_{MODEL_RATIO}/resnet/{VERSION_RESNET}"

        VERSION_CONVNEXT = "v4"
        MODEL_CONVNEXT_PATH = f"/work/dldevel/trieba/bachelor_thesis/results/model_{MODEL_RATIO}/convnext/{VERSION_CONVNEXT}"

        SUPERCLASS_MAP = "/work/dldevel/trieba/bachelor_thesis/training_data/misc/superclasses_mapping.json"
        DATASET_PATH = "/work/dldevel/trieba/bachelor_thesis/training_data/eval_dataset"
        OUTPUT_PATH = f"/work/dldevel/trieba/bachelor_thesis/results/evaluation/{MODEL_RATIO}/{VERSION}"

        VARIANTS = [
            "original", "only_bg_b", "only_bg_t", "only_bg_noise",
            "no_fg", "no_fg_noise", "only_fg", "only_fg_noise",
            "mixed_same", "mixed_rand"
        ]
        BATCH_SIZE = 128

        with open(SUPERCLASS_MAP, "r") as f:
            superclasses = json.load(f)

        with open(f"{MODEL_RESNET_PATH}/maps_to_idx.json", "r") as f:
            maps_to_idx = json.load(f) 

        NUM_CLASSES = len(maps_to_idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sup_ids = {s: i for i, s in enumerate(sorted({superclasses[c] for c in maps_to_idx}))}
        sup_of_idx = torch.full((NUM_CLASSES,), -1, dtype=torch.long)
        for cls_name, idx in maps_to_idx.items():
            sup_of_idx[int(idx)] = sup_ids[superclasses[cls_name]]
        assert (sup_of_idx >= 0).all()
        sup_of_idx = sup_of_idx.to(device)

        counts = torch.bincount(sup_of_idx.cpu())
        print(f"Superclass chance level: {100 * ((counts / counts.sum()) ** 2).sum().item():.1f}%")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        def load_models():
            resnet = models.resnet50(weights=None)
            resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
            resnet.load_state_dict(torch.load(f"{MODEL_RESNET_PATH}/best_checkpoint.pth", map_location=device))
            resnet.to(device)
            resnet.eval()

            convnext = models.convnext_tiny(weights=None)
            convnext.classifier[2] = torch.nn.Linear(convnext.classifier[2].in_features, NUM_CLASSES)
            convnext.load_state_dict(torch.load(f"{MODEL_CONVNEXT_PATH}/best_checkpoint.pth", map_location=device))
            convnext.to(device)
            convnext.eval()

            return {"ResNet50": resnet, "ConvNeXt": convnext}


        def build_label_remap(dataset_classes):
            remap = []
            for folder_name in dataset_classes:
                if folder_name not in maps_to_idx:
                    raise KeyError(
                        f"Folder '{folder_name}' has no entry in maps_to_idx.json -- "
                        "cannot determine its true training label index."
                    )
                remap.append(int(maps_to_idx[folder_name]))
            return torch.tensor(remap, dtype=torch.long, device=device)


        def superclass_breakdown(per_class):
            agg = defaultdict(lambda: {"correct": 0, "total": 0})
            for cls_name, stats in per_class.items():
                sup = superclasses.get(cls_name)
                if sup is None:
                    continue
                agg[sup]["correct"] += stats["correct"]
                agg[sup]["total"] += stats["total"]
            return {
                sup: {
                    **vals,
                    "accuracy": round(100 * vals["correct"] / vals["total"], 2) if vals["total"] else 0.0,
                }
                for sup, vals in agg.items()
            }


        def evaluate_model(model, dataloader, label_remap, dataset_classes):
            correct = 0
            sup_correct = 0  
            total = 0
            total_confidence = 0.0
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            class_sup_correct = defaultdict(int)

            with torch.no_grad():
                for images, folder_labels in dataloader:
                    images = images.to(device)
                    folder_labels = folder_labels.to(device)

                    true_labels = label_remap[folder_labels]

                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted = outputs.argmax(dim=1)

                    correct_mask = predicted == true_labels
                    correct += correct_mask.sum().item()
                    sup_mask = sup_of_idx[predicted] == sup_of_idx[true_labels]
                    sup_correct += sup_mask.sum().item()
                    total += true_labels.size(0)

                    batch_idx = torch.arange(true_labels.size(0), device=device)
                    total_confidence += probabilities[batch_idx, true_labels].sum().item()

                    for folder_idx, is_correct, is_sup in zip(
                        folder_labels.cpu().tolist(), correct_mask.cpu().tolist(), sup_mask.cpu().tolist()
                    ):
                        cls_name = dataset_classes[folder_idx]
                        class_total[cls_name] += 1
                        if is_correct:
                            class_correct[cls_name] += 1
                        if is_sup:
                            class_sup_correct[cls_name] += 1 

            accuracy = 100 * correct / total
            sup_accuracy = 100 * sup_correct / total
            avg_confidence = 100 * total_confidence / total
            per_class = {
                cls: {
                    "correct": class_correct[cls],
                    "total": class_total[cls],
                    "accuracy": round(100 * class_correct[cls] / class_total[cls], 2) if class_total[cls] else 0.0,
                    "superclass_correct": class_sup_correct[cls],
                    "superclass_accuracy": round(100 * class_sup_correct[cls] / class_total[cls], 2) if class_total[cls] else 0.0,
                }
                for cls in class_total
            }
            return accuracy, sup_accuracy, avg_confidence, per_class


        models_dict = load_models()
        results = {}
        reference_classes = None

        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            results[model_name] = {}

            for variant in VARIANTS:
                variant_dir = os.path.join(DATASET_PATH, variant)
                if not os.path.isdir(variant_dir):
                    print(f"  Warning: variant directory not found, skipping: {variant_dir}")
                    continue

                dataset = datasets.ImageFolder(root=variant_dir, transform=transform)

                if reference_classes is None:
                    reference_classes = set(dataset.classes)
                elif set(dataset.classes) != reference_classes:
                    missing = reference_classes - set(dataset.classes)
                    extra = set(dataset.classes) - reference_classes
                    print(
                        f"  Warning: class mismatch in '{variant}'. "
                        f"Missing: {missing or 'none'}, Unexpected: {extra or 'none'}"
                    )

                label_remap = build_label_remap(dataset.classes)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=8, 
                    pin_memory=True,
                    persistent_workers=True
                )

                acc, sup_acc, conf, per_class = evaluate_model(model, dataloader, label_remap, dataset.classes)
                sup_breakdown = superclass_breakdown(per_class)

                results[model_name][variant] = {
                    "Accuracy": acc,
                    "SuperclassAccuracy": sup_acc,
                    "Confidence": conf,
                    "PerClass": per_class,
                    "PerSuperclass": sup_breakdown,
                }

                print(f"  {variant.ljust(14)} -> Accuracy: {acc:.2f}% | Avg True-Class Confidence: {conf:.2f}%")


            if "mixed_same" in results[model_name] and "mixed_rand" in results[model_name]:
                same_acc = results[model_name]["mixed_same"]["Accuracy"]
                rand_acc = results[model_name]["mixed_rand"]["Accuracy"]
                bg_gap = round(same_acc - rand_acc, 2)
                results[model_name]["background_gap"] = bg_gap
                print(f"  Background-Gap (Mixed-Same - Mixed-Rand): {bg_gap:.2f} pts")

                same_sup_acc = results[model_name]["mixed_same"]["SuperclassAccuracy"]
                rand_sup_acc = results[model_name]["mixed_rand"]["SuperclassAccuracy"]
                results[model_name]["background_gap_superclass"] = round(same_sup_acc - rand_sup_acc, 2)
                print(f"  Background-Gap, superclass level: {same_sup_acc - rand_sup_acc:.2f} pts")

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        with open(f"{OUTPUT_PATH}/eval_dataset_results.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nEvaluation Complete. Results saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    mp.set_start_method("fork", force=True)
    main()