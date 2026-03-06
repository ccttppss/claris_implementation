import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from accelerate.logging import get_logger

logger = get_logger(__name__)

class MVTecADDatasetAugmented(Dataset):

    def __init__(self, root_dir, tokenizer, info_map_data, size=512,
                 category=None, split='train'):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.split = split
        self.base_data = []

        category_path = self.root_dir / category
        if not category_path.is_dir():
            raise ValueError(f"Category '{category}' not found.")

        paired_data = []
        best_normal_dir = category_path / "best"
        test_dir = category_path / "test"
        normal_map_root = category_path / "best_normal"
        ground_truth_root = category_path / "ground_truth"

        if not best_normal_dir.is_dir():
            raise FileNotFoundError(f"Paired normal images not found in {best_normal_dir}")

        defect_types = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        for defect_path in defect_types:
            for defect_img_path in defect_path.glob("*.png"):
                paired_normal_path = best_normal_dir / defect_path.name / defect_img_path.name
                mask_path = ground_truth_root / defect_path.name / f"{defect_img_path.stem}_mask.png"
                normal_map_path = normal_map_root / defect_path.name / defect_img_path.name

                if paired_normal_path.exists() and mask_path.exists() and normal_map_path.exists():
                    key = f"{category}_{defect_path.name}"
                    if key not in info_map_data:
                        continue

                    placeholder = " ".join(info_map_data[key]["placeholder_tokens"])
                    caption = f"a photo of a {category} with {placeholder} defect"
                    tok = tokenizer(caption, max_length=tokenizer.model_max_length,
                                    padding="max_length", truncation=True)
                    input_ids = torch.tensor(tok.input_ids)

                    placeholder_ids = set(tokenizer.convert_tokens_to_ids(
                        info_map_data[key]["placeholder_tokens"]))
                    token_indices = torch.tensor([
                        i if tid in placeholder_ids else -100
                        for i, tid in enumerate(input_ids)
                    ])

                    paired_data.append({
                        "key": key,
                        "image_path_target": defect_img_path,
                        "image_path_input": paired_normal_path,
                        "mask_path": mask_path,
                        "normal_map_path": normal_map_path,
                        "input_ids": input_ids,
                        "token_indices": token_indices,
                    })

        if not paired_data:
            raise ValueError(f"No defect-normal pairs found for category '{category}'")

        from collections import defaultdict
        groups = defaultdict(list)
        for item in paired_data:
            groups[item["key"]].append(item)

        rng = random.Random(42)
        train_items, val_items, test_items = [], [], []

        for k, g_items in groups.items():
            rng.shuffle(g_items)
            n = len(g_items)

            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.10))

            if n_train == 0 and n > 0:
                n_train = 1
            if n_train >= n:
                n_train = max(1, n - 1)
                n_val = 0
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)

            train_items.extend(g_items[:n_train])
            val_items.extend(g_items[n_train:n_train + n_val])
            test_items.extend(g_items[n_train + n_val:])

        if split == 'train':
            self.base_data = train_items
        elif split == 'val':
            self.base_data = val_items
        elif split == 'test':
            self.base_data = test_items
        else:
            self.base_data = test_items

        logger.info(f"[{category}] Stratified Split -> TRAIN: {len(train_items)} (70%) | VAL: {len(val_items)} (10%) | TEST: {len(test_items)} (20%) | Selected = {split.upper()} ({len(self.base_data)} items)")

        self.image_transforms = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size)
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size)
        ])

    def __len__(self):
        if self.split == 'train':
            return len(self.base_data) * 8
        return len(self.base_data)

    def __getitem__(self, idx):
        if self.split == 'train':
            base_idx, aug_type = divmod(idx, 8)
        else:
            base_idx, aug_type = idx, 0

        item = self.base_data[base_idx]

        target_img = Image.open(item["image_path_target"]).convert("RGB")
        input_img = Image.open(item["image_path_input"]).convert("RGB")

        target_img = self.image_transforms(target_img)
        input_img = self.image_transforms(input_img)

        tensor_img_target = TF.to_tensor(target_img) * 2 - 1
        tensor_img_input = TF.to_tensor(input_img) * 2 - 1

        mask = Image.open(item["mask_path"]).convert("L")
        mask = self.mask_transforms(mask)
        mask_t = TF.to_tensor(mask)

        nm = Image.open(item["normal_map_path"]).convert("RGB")
        nm = self.image_transforms(nm)
        control = TF.to_tensor(nm) * 2 - 1

        if aug_type > 0:
            tensors_to_augment = [tensor_img_target, tensor_img_input, mask_t, control]

            if aug_type == 1:
                augmented_tensors = [TF.hflip(t) for t in tensors_to_augment]
            elif aug_type == 2:
                augmented_tensors = [TF.vflip(t) for t in tensors_to_augment]
            elif aug_type == 3:
                augmented_tensors = [TF.rotate(t, 90) for t in tensors_to_augment]
            elif aug_type == 4:
                augmented_tensors = [TF.rotate(t, 180) for t in tensors_to_augment]
            elif aug_type == 5:
                augmented_tensors = [TF.rotate(t, 270) for t in tensors_to_augment]
            elif aug_type == 6:
                t_rot = [TF.rotate(t, 90) for t in tensors_to_augment]
                augmented_tensors = [TF.hflip(t) for t in t_rot]
            elif aug_type == 7:
                t_rot = [TF.rotate(t, 90) for t in tensors_to_augment]
                augmented_tensors = [TF.vflip(t) for t in t_rot]

            tensor_img_target, tensor_img_input, mask_t, control = augmented_tensors

        return {
            "pixel_values_target": tensor_img_target,
            "pixel_values_input": tensor_img_input,
            "pixel_values_mask": mask_t,
            "control_image": control,
            "input_ids": item["input_ids"],
            "token_indices": item["token_indices"],
        }
