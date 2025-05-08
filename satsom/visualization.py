import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from satsom.model import SatSOM


def create_satsom_image(model: SatSOM, output_path: str, img_width, img_height):
    if len(model.params.grid_shape) != 2:
        raise NotImplementedError(
            "`create_satsom_image` not yet implemented for n-dimensional case"
        )

    m, n = model.params.grid_shape
    weights = model.weights.data
    prototypes = weights.view(m, n, img_height, img_width).cpu()

    # 1) most‑activated label per neuron
    label_idxs = torch.argmax(model.labels.data, dim=1).view(m, n).cpu().numpy()
    num_labels = model.params.output_dim

    # 2) categorical colormap
    if num_labels <= 10:
        cmap = plt.get_cmap("tab10", num_labels)
    elif num_labels <= 20:
        cmap = plt.get_cmap("tab20", num_labels)
    else:
        cmap = plt.get_cmap("hsv", num_labels)

    sat_cmap = plt.get_cmap("magma")

    # 3) saturations [0..1]
    sats = (model.params.initial_lr - model.learning_rates).view(m, n, 1).cpu()
    sats = (sats - sats.min()) / (sats.max() - sats.min() + 1e-6)
    sats = sats.squeeze(2).numpy()  # (m,n)

    # grid
    stride = 1
    rows = list(range(0, m, stride))
    cols = list(range(0, n, stride))
    R, C = len(rows), len(cols)
    H, W = img_height, img_width

    # two canvases: full‐tint & sat‐tint
    full_tint = np.zeros((R * H, C * W, 3))
    sat_tint = np.zeros((R * H, C * W, 3))

    for ri, i in enumerate(rows):
        for ci, j in enumerate(cols):
            # normalize grayscale
            img = prototypes[i, j].numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            base = img[..., None]  # (H,W,1)

            # label color
            lbl = label_idxs[i, j]
            color = np.array(cmap(lbl)[:3])[None, None, :]  # (1,1,3)

            # apply some color also to the black regions
            eps = 0.5
            whitened_base = base * (1 - eps) + eps

            y0, y1 = ri * H, (ri + 1) * H
            x0, x1 = ci * W, (ci + 1) * W

            # full tint: pure color blend
            full_tint[y0:y1, x0:x1, :] = whitened_base * color

            sat = sats[i, j]
            color = np.array(sat_cmap(sat)[:3])[None, None, :]
            sat_tint[y0:y1, x0:x1, :] = color

    # plot both tinted panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    ax1.imshow(full_tint, interpolation="nearest")
    ax1.set_title("Prototypes (class-tinted)")
    ax1.axis("off")

    ax2.imshow(sat_tint, interpolation="nearest")
    ax2.set_title("Saturations (normalized)")
    ax2.axis("off")

    # Add a normalized saturation colorbar
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=sat_cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
