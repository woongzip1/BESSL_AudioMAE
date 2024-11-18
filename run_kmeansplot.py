import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import entropy
from run_kmeans import CustomDataset, AudioMAEEncoder
from utils import SmallDataset

# Assuming the necessary classes like CustomDataset, AudioMAEEncoder are imported

# Load Dataset
dataset = CustomDataset(
    path_dir_wb=["/home/woongzip/FSD50K_16kHz", "/home/woongzip/MUSDB18_HQ_16kHz_mono"],
    path_dir_nb=["/home/woongzip/FSD50K_16kHz_codec", "/home/woongzip/MUSDB18_MP3_8k"],
    seg_len=9.6,
    mode='train'
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=32, prefetch_factor=True)
# subdataset = SmallDataset(dataloader, 200)
# dataloader = DataLoader(subdataset, batch_size=128, shuffle=True)

# Load Feature Encoder Model
model = AudioMAEEncoder(visualize=False).eval().cuda()
model.load_weights()

# Function to quantize input and calculate entropy for each band
def quantize_and_calculate_entropy(input_tensor, kmeans_models):
    # T x 6144
    # B, T, _ = input_tensor.shape

    num_patches = len(kmeans_models)
    patch_size = 768  # Each patch has 768 dimensions
    band_entropies = []

    for idx in range(num_patches):
        patch_start = patch_size * idx
        patch_end = patch_start + patch_size

        # Extract the patch-specific embeddings
        patch_embeddings = input_tensor[:, patch_start:patch_end]  # T x 768

        # Use k-means model to quantize the patch embeddings
        kmeans_model = kmeans_models[idx]
        cluster_labels = kmeans_model.predict(patch_embeddings.cpu().numpy())  # (B*T)

        # Calculate entropy for the current patch (band)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        prob_dist = counts / np.sum(counts)
        band_entropy = entropy(prob_dist, base=2)
        band_entropies.append(band_entropy)

        print(f"Patch {idx} entropy: {band_entropy:.3f}")

    return band_entropies

# Processing the DataLoader
embs = []
for spec, name in tqdm(dataloader, desc="Extracting embeddings"):
    with torch.no_grad():
        spec = spec.cuda()
        features = model(spec.unsqueeze(1))
        embs.append(features.cpu())  # Immediately move to CPU memory
        torch.cuda.empty_cache()  # Clear GPU cache to free up memory

# Combine embeddings
data_tensor = torch.cat(embs, dim=0)
print(f"Data tensor shape:", data_tensor.shape)
# B x 64 x 6144

# Reshape embeddings for k-means processing
embeddings = data_tensor.reshape(-1, data_tensor.shape[-1])
print(f"Embeddings shape:", embeddings.shape)
# FRAMES x 6144 (8 patches)

# Load K-means models and calculate entropy for each band
for N in [4, 8, 16, 32, 64, 128, 256, 512]:
    kmeans_model_path = f"/home/woongzip/BESSL/kmeans/K{N}_MAE.pkl"

    with open(kmeans_model_path, 'rb') as file:
        kmeans_models = pickle.load(file)

    # Embeddings: T x 6144
    band_entropies = quantize_and_calculate_entropy(embeddings, kmeans_models)

    # Plot the entropy values for each band
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(band_entropies)), band_entropies, tick_label=[f'Band {i}' for i in range(len(band_entropies))])
    plt.xlabel('Band Index')
    plt.ylabel('Entropy')
    plt.title(f'Entropy of Each Band (K={N})')
    plt.grid(axis='y')
    plt.savefig(f'entropy_plot_K{N}.png', format='png', dpi=300)
    plt.show()

    print(f"Entropy plot saved as entropy_plot_K{N}.png")
