import numpy as np
import csv
from collections import defaultdict
import datetime as dt
import os
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import yaml
import zarr
import matplotlib.pyplot as plt
import seaborn as sns


def read_yaml_class_mapping(country):
    return yaml.load(open("denmark_class_mapping.yml"), Loader=yaml.FullLoader)

def get_code_to_class(country, combine_spring_and_winter=False):
    class_mapping = read_yaml_class_mapping(country)

    code_to_class = {}
    for cls in class_mapping.keys():
        codes = class_mapping[cls]
        if codes is None:
            continue
        if 'spring' in codes and 'winter' in codes:
            if combine_spring_and_winter:
                combined = {**(codes['spring'] if codes['spring'] is not None else {}), **(codes['winter'] if codes['winter'] is not None else {})}
                code_to_class.update({code: cls for code in combined})
            else:
                if codes['spring'] is not None:
                    code_to_class.update({code: f'spring_{cls}' for code in codes['spring'].keys()})
                if codes['winter'] is not None:
                    code_to_class.update({code: f'winter_{cls}' for code in codes['winter'].keys()})
        else:
            code_to_class.update({code: cls for code in codes})
    return code_to_class

def get_classes(*countries, method=set.union, combine_spring_and_winter=False):
    class_sets = []
    for country in countries:
        code_to_class = get_code_to_class(country, combine_spring_and_winter)
        class_sets.append(set(code_to_class.values()))

    classes = sorted(list(method(*class_sets)))
    return classes

def get_codification_table(country):
    codification_table = os.path.join('class_mapping', f'{country}_codification_table.csv')
    with open(codification_table, newline='') as f:
        delimiter = ';' if country in ['denmark', 'austria'] else ','
        crop_codes = csv.reader(f, delimiter=delimiter)
        crop_codes = {x[0]: x[1] for x in crop_codes}  # crop_code: (name, group)
    return crop_codes

def plot_distribution(label_names, counts, save_path):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(label_names, counts, color='skyblue', edgecolor='black')

    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, str(count),
                ha='center', va='bottom', fontsize=10)

    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Classes in the Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg", dpi=300)
    plt.show()

class PixelSetData(Dataset):
    def __init__(
        self,
        data_root,
        dataset_name,
        classes,
        transform=None,
        indices=None,
        with_extra=False,
    ):
        super(PixelSetData, self).__init__()

        self.folder = os.path.join(data_root, dataset_name)
        self.dataset_name = dataset_name  # country/tile/year
        self.country = dataset_name.split("/")[-3]
        self.tile = dataset_name.split("/")[-2]
        self.data_folder = os.path.join(self.folder, "data")
        self.meta_folder = os.path.join(self.folder, "meta")
        self.transform = transform
        self.with_extra = with_extra

        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.samples, self.metadata = self.make_dataset(
            self.data_folder, self.meta_folder, self.class_to_idx, indices, self.country
        )

        self.dates = self.metadata["dates"]
        self.date_positions = self.days_after(self.metadata["start_date"], self.dates)
        self.date_indices = np.arange(len(self.date_positions))

    def get_shapes(self):
        return [
            (len(self.dates), 10, parcel["n_pixels"])
            for parcel in self.metadata["parcels"]
        ]

    def get_labels(self):
        return np.array([x[2] for x in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, parcel_idx, y, extra = self.samples[index]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": index,
            "parcel_index": parcel_idx,  # mapping to metadata
            "pixels": pixels,
            "valid_pixels": np.ones(
                (pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "positions": np.array(self.date_positions),
            "extra": np.array(extra),
            "label": y,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_dataset(self, data_folder, meta_folder, class_to_idx, indices, country):
        metadata = pkl.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))

        instances = []
        new_parcel_metadata = []

        code_to_class_name = get_code_to_class(country)

        unknown_crop_codes = set()

        for parcel_idx, parcel in enumerate(metadata["parcels"]):
            if indices is not None:
                if not parcel_idx in indices:
                    continue
            crop_code = parcel["label"]
            if country == "austria":
                crop_code = int(crop_code)
            parcel_path = os.path.join(data_folder, f"{parcel_idx}.zarr")
            if not os.path.exists(parcel_path):
                continue 
            if crop_code not in code_to_class_name:
                unknown_crop_codes.add(crop_code)
            class_name = code_to_class_name.get(crop_code, "unknown")
            class_index = class_to_idx.get(class_name, class_to_idx["unknown"])
            extra = parcel['geometric_features']

            item = (parcel_path, parcel_idx, class_index, extra)
            instances.append(item)
            new_parcel_metadata.append(parcel)

        for crop_code in unknown_crop_codes:
            print(
                f"Parcels with crop code {crop_code} was not found in .yml class mapping and was assigned to unknown."
            )

        metadata["parcels"] = new_parcel_metadata

        assert len(metadata["parcels"]) == len(instances)

        return instances, metadata

    def days_after(self, start_date, dates):
        def parse(date):
            d = str(date)
            return int(d[:4]), int(d[4:6]), int(d[6:])

        def interval_days(date1, date2):
            return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)

        date_positions = [interval_days(d, start_date) for d in dates]
        return date_positions

    def get_unknown_labels(self):
        """
        Reports the categorization of crop codes for this dataset
        """
        class_count = defaultdict(int)
        class_parcel_size = defaultdict(float)
        # metadata = pkl.load(open(os.path.join(self.meta_folder, 'metadata.pkl'), 'rb'))
        metadata = self.metadata
        for meta in metadata["parcels"]:
            class_count[meta["label"]] += 1
            class_parcel_size[meta["label"]] += meta["n_pixels"]

        class_avg_parcel_size = {
            cls: total_px / class_count[cls]
            for cls, total_px in class_parcel_size.items()
        }

        code_to_class_name = get_code_to_class(self.country)
        codification_table = get_codification_table(self.country)
        unknown = []
        known = defaultdict(list)
        for code, count in class_count.items():
            avg_pixels = class_avg_parcel_size[code]
            if self.country == "denmark":
                code = int(code)
            code_name = codification_table[str(code)]
            if code in code_to_class_name:
                known[code_to_class_name[code]].append(
                    (code, code_name, count, avg_pixels)
                )
            else:
                unknown.append((code, code_name, count, avg_pixels))

        print("\nCategorized crop codes:")
        for class_name, codes in known.items():
            total_parcels = sum(x[2] for x in codes)
            avg_parcel_size = sum(x[3] for x in codes) / len(codes)
            print(f"{class_name} (n={total_parcels}, avg size={avg_parcel_size:.3f}):")
            codes = reversed(sorted(codes, key=lambda x: x[2]))
            for code, code_name, count, avg_pixels in codes:
                print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")
        unknown = reversed(sorted(unknown, key=lambda x: x[2]))
        print("\nUncategorized crop codes:")
        for code, code_name, count, avg_pixels in unknown:
            print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")

class WrappedPixelSubset(Dataset):
    def __init__(self, base_dataset, indices, sample_pixels=None, is_validation=False):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.sample_pixels = sample_pixels
        self.is_validation = is_validation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]

        # Apply pixel sampling only if training
        if not self.is_validation and self.sample_pixels is not None:
            pixels = sample["pixels"]  # (T, C, S)
            total_pixels = pixels.shape[-1]

            if total_pixels >= self.sample_pixels:
                sampled_indices = np.random.choice(total_pixels, self.sample_pixels, replace=False)
            else:
                sampled_indices = np.random.choice(total_pixels, self.sample_pixels, replace=True)

            pixels = pixels[:, :, sampled_indices]
            sample["pixels"] = pixels
            sample["valid_pixels"] = np.ones((pixels.shape[0], pixels.shape[-1]), dtype=np.float32)

        return sample
    
class NormalizePerBand:
    def __init__(self, mean, std):
        # of shape (C, )
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # of shape (T, C, S)
        pixels = sample["pixels"]
        # broadcast to shape (1, C, 1)
        mean = self.mean[None, :, None]
        std = self.std[None, :, None]
        sample["pixels"] = (pixels - mean) / std

def compute_band_stats_from_dl(dataloader):
    per_band_sum, per_band_sum_sq, count = 0, 0, 0

    for batch in dataloader:
        # (B, T, C, S)
        pixels = batch["pixels"].float()
        # flatten to (B*T*S, c)
        pixels = pixels.view(-1, pixels.shape[2])
        per_band_sum += pixels.sum(dim=0)
        per_band_sum_sq += (pixels ** 2).sum(dim=0)
        count += pixels.shape[0]

    per_band_mean = per_band_sum / count
    per_band_std = np.sqrt(per_band_sum_sq / count - per_band_mean ** 2)
    return per_band_mean, per_band_std



def get_dataloaders_for_cv(dataset, n_splits=5, sample_pixels=32, batch_size=8):
    indices = np.arange(len(dataset))
    folds = np.array_split(indices, n_splits)
    dataloaders = []

    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])

        train_subset = WrappedPixelSubset(dataset, train_idx, sample_pixels=sample_pixels, is_validation=False)
        val_subset = WrappedPixelSubset(dataset, val_idx, sample_pixels=None, is_validation=True)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        dataloaders.append((train_loader, val_loader))

    return dataloaders

def normalize_batch(batch, mean, std):
    pixels = batch["pixels"].float()
    # broadcast to (1, C, 1)
    mean = mean[None, :, None]
    std = std[None, :, None]
    batch["pixels"] = (pixels - mean) / (std + 1e-6)
    batch["extra"] = batch["extra"].float()
    return batch

def plot_metrics_shaded(per_phase_metric, y_label, title, save_path):
    # of shape (num_folds, num_epochs)
    metrics_array = np.array(per_phase_metric)  
    mean_metric = np.mean(metrics_array, axis=0)
    std_metric = np.std(metrics_array, axis=0)

    epochs = np.arange(1, len(mean_metric) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_metric, label=f'Mean {y_label}')
    plt.fill_between(epochs, mean_metric - std_metric, mean_metric + std_metric, alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg", dpi=300)
    plt.show()

def train_one_epoch(model, optimizer, train_dl, per_channel_means, per_channel_stds, device):
    criterion = CrossEntropyLoss()
    curr_loss = 0.
    model.to(device)
    model.train()
    for i, batch in enumerate(train_dl):
        batch = normalize_batch(batch, per_channel_means, per_channel_stds)
        batch["pixels"] = batch["pixels"].to(device)
        batch["valid_pixels"] = batch["valid_pixels"].to(device)
        batch["positions"] = batch["positions"].to(device)
        batch["extra"] = batch["extra"].to(device)
        batch["label"] = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(pixels=batch["pixels"], mask=batch["valid_pixels"], positions=batch["positions"], extra=batch["extra"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()
    return curr_loss / len(train_dl)

def validate(model, val_dl, per_channel_means, per_channel_stds, device):
    criterion = CrossEntropyLoss()
    curr_loss, all_preds, all_labels = 0., [], []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(val_dl):
            batch = normalize_batch(batch, per_channel_means, per_channel_stds)
            batch["pixels"] = batch["pixels"].to(device)
            batch["valid_pixels"] = batch["valid_pixels"].to(device)
            batch["positions"] = batch["positions"].to(device)
            batch["extra"] = batch["extra"].to(device)
            batch["label"] = batch["label"].to(device)

            logits = model(pixels=batch["pixels"], mask=batch["valid_pixels"], positions=batch["positions"], extra=batch["extra"])
            loss = criterion(logits, batch["label"])
            curr_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch["label"].cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return curr_loss / len(val_dl), all_labels, all_preds

def calculate_val_metrics(all_preds, all_labels, conf_matrix=False):
    accuracy = accuracy_score(all_labels, all_preds)
    macro_pres = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_pres = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    if conf_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, macro_pres, macro_rec, macro_f1, weighted_pres, weighted_rec, weighted_f1, cm
    else:
        return accuracy, macro_pres, macro_rec, macro_f1, weighted_pres, weighted_rec, weighted_f1


def plot_metrics_and_heatmaps(metrics_dict, heatmaps, heatmap_titles=None, title='Metrics & Confusion Matrices'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_metrics = len(metrics_dict)
    num_heatmaps = len(heatmaps)
    total_plots = num_metrics + num_heatmaps

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 18))
    axs = axs.flatten()

    fold_ids = [f"Fold {i+1}" for i in range(len(next(iter(metrics_dict.values()))))]

    # plot bar charts for each metric
    for i, (metric_name, scores) in enumerate(metrics_dict.items()):
        ax = axs[i]
        bars = ax.bar(fold_ids, scores, color='mediumseagreen', edgecolor='black')
        ax.set_title(metric_name)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y')
        ax.bar_label(bars, fmt="%.2f", padding=3)

    # plot heatmaps for each fold
    for j, cm in enumerate(heatmaps):
        ax = axs[num_metrics + j]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, square=True)
        if heatmap_titles:
            ax.set_title(heatmap_titles[j])
        else:
            ax.set_title(f'Confusion Matrix Fold {j+1}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Hide any unused subplots
    for k in range(total_plots, len(axs)):
        axs[k].axis('off')

    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("best_metrics_and_cms.png", dpi=300)
    plt.savefig("best_metrics_and_cms.svg", dpi=300)
    plt.show()


