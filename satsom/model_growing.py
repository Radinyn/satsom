import torch
import torch.nn as nn
from dataclasses import replace
from satsom.model import SatSOM, SatSOMParameters


class GrowingSatSOM(nn.Module):
    def __init__(
        self,
        params: SatSOMParameters,
        naive: bool = False,
        centering: bool = True,
        radius_threshold_scale: float = 0.3,
    ):
        super().__init__()
        self.initial_params = params
        self.satsom = SatSOM(params)
        self.naive = naive
        self.centering = centering
        self.growth_increment = min(params.grid_shape)
        self.epsilon = 1e-4
        self.radius_threshold_scale = radius_threshold_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.satsom(x)

    def step(self, x: torch.Tensor, label: torch.Tensor):
        self.grow_map(x)
        self.satsom.step(x, label)

    def _get_active_mask_and_bounds(
        self,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Returns 2D boolean mask of active neurons and bounding box (r_min, r_max, c_min, c_max)."""
        H, W = self.satsom.params.grid_shape

        # Calculate saturation
        saturation = (
            self.satsom.params.initial_lr - self.satsom.learning_rates
        ) / self.satsom.params.initial_lr

        # Identify active neurons (> epsilon)
        active_1d = saturation > self.epsilon
        active_2d = active_1d.view(H, W)

        if not active_2d.any():
            # If nothing active, return center pixel or full bounds
            return active_2d, (0, H, 0, W)

        # Find bounds
        rows, cols = torch.where(active_2d)
        r_min, r_max = rows.min().item(), rows.max().item()
        c_min, c_max = cols.min().item(), cols.max().item()

        return active_2d, (r_min, r_max, c_min, c_max)

    def _transfer_state(
        self,
        new_som: SatSOM,
        offset: tuple[int, int],
        old_bounds: tuple[int, int, int, int],
    ):
        """Transfers weights from current satsom to new_som at specified offset."""
        r_min, r_max, c_min, c_max = old_bounds
        r_off, c_off = offset

        H_old, W_old = self.satsom.params.grid_shape

        # With the current usage they are always equal to old dimensions
        H_new, W_new = new_som.params.grid_shape

        # Reshape current state to 2D grid for slicing
        old_w = self.satsom.weights.view(H_old, W_old, -1)
        old_lr = self.satsom.learning_rates.view(H_old, W_old)
        old_sig = self.satsom.sigmas.view(H_old, W_old)
        old_lbl = self.satsom.labels.view(H_old, W_old, -1)

        # Prepare target state
        new_w = new_som.weights.view(H_new, W_new, -1)
        new_lr = new_som.learning_rates.view(H_new, W_new)
        new_sig = new_som.sigmas.view(H_new, W_new)
        new_lbl = new_som.labels.view(H_new, W_new, -1)

        # Determine slice dimensions
        h_slice = r_max - r_min + 1
        w_slice = c_max - c_min + 1

        # Copy data
        with torch.no_grad():
            # 1. New (Target) Index Calculation with Toroidal Wrapping
            # Indices for the rows/cols in the new map, wrapped around H_new/W_new
            row_indices_new = (
                torch.arange(r_off, r_off + h_slice, device=new_w.device) % H_new
            ).long()
            col_indices_new = (
                torch.arange(c_off, c_off + w_slice, device=new_w.device) % W_new
            ).long()

            # Create a 2D meshgrid of indices for the new map (where to write)
            r_idx, c_idx = torch.meshgrid(
                row_indices_new, col_indices_new, indexing="ij"
            )

            # 2. Old (Source) Index Calculation
            # Indices for the rows/cols in the old map (what to read)
            row_indices_old = torch.arange(r_min, r_max + 1, device=new_w.device).long()
            col_indices_old = torch.arange(c_min, c_max + 1, device=new_w.device).long()

            # Create a 2D meshgrid of indices for the old map (where to read from)
            r_old_idx, c_old_idx = torch.meshgrid(
                row_indices_old, col_indices_old, indexing="ij"
            )

            # 3. Use advanced indexing (r_idx, c_idx) to copy data

            # Read slices from old map
            old_slice_w = old_w[r_old_idx, c_old_idx]
            old_slice_lr = old_lr[r_old_idx, c_old_idx]
            old_slice_sig = old_sig[r_old_idx, c_old_idx]
            old_slice_lbl = old_lbl[r_old_idx, c_old_idx]

            # Write to the new map using the wrapped indices
            new_w[r_idx, c_idx] = old_slice_w
            new_lr[r_idx, c_idx] = old_slice_lr
            new_sig[r_idx, c_idx] = old_slice_sig
            new_lbl[r_idx, c_idx] = old_slice_lbl

            new_som.weights.data = new_w.reshape(-1, new_som.params.input_dim)
            new_som.learning_rates.data = new_lr.reshape(-1)
            new_som.sigmas.data = new_sig.reshape(-1)
            new_som.labels.data = new_lbl.reshape(-1, new_som.params.output_dim)

    def center_map(self):
        """Centers the active neuron bounding box within the current grid."""
        _, (r_min, r_max, c_min, c_max) = self._get_active_mask_and_bounds()
        H, W = self.satsom.params.grid_shape

        # Dimensions of the active box
        box_h = r_max - r_min + 1
        box_w = c_max - c_min + 1

        # Calculate target top-left to center this box
        target_r = (H - box_h) // 2
        target_c = (W - box_w) // 2

        # If already centered, skip
        if target_r == r_min and target_c == c_min:
            return

        # Create identical new SOM structure
        new_som = SatSOM(self.satsom.params).to(self.satsom.weights.device)

        # Transfer active neurons to the calculated center
        self._transfer_state(
            new_som, (target_r, target_c), (r_min, r_max, c_min, c_max)
        )
        self.satsom = new_som

    def grow_map(self, x: torch.Tensor):
        """Performs centering (if enabled) then grows the map based on strategy."""
        if self.centering:
            self.center_map()

        H, W = self.satsom.params.grid_shape

        # Determine growth parameters
        new_H, new_W = H, W
        offset_r, offset_c = 0, 0

        # Check if BMU neighborhood significantly extends outside map
        bmu_idx = self.satsom.find_bmu(x.unsqueeze(0)).item()
        bmu_r, bmu_c = divmod(bmu_idx, W)

        # Get sigma for BMU to estimate neighborhood radius
        sigma = self.satsom.sigmas[bmu_idx].item()

        # Heuristic: if edge is within a certain fraction of sigma, the neighborhood is "cut off"
        radius_threshold = max(self.radius_threshold_scale * sigma, 1.0)

        pad = self.growth_increment

        # Check boundaries based on radius
        grow_top = bmu_r < radius_threshold
        grow_bottom = (H - 1 - bmu_r) < radius_threshold
        grow_left = bmu_c < radius_threshold
        grow_right = (W - 1 - bmu_c) < radius_threshold

        if not (grow_top or grow_bottom or grow_left or grow_right):
            return  # Neighborhood contained within map, no growth

        if self.naive:
            # Expand by min dimension in ALL directions
            pad = self.growth_increment
            new_H = H + 2 * pad
            new_W = W + 2 * pad
            # We map the entire old grid to the center of new grid
            offset_r = pad
            offset_c = pad
            source_bounds = (0, H - 1, 0, W - 1)  # Copy full old map
        else:
            # Smart Growing: Direction dependent
            if grow_top:
                new_H += pad
                offset_r += pad  # Shift old map down
            if grow_bottom:
                new_H += pad
            if grow_left:
                new_W += pad
                offset_c += pad  # Shift old map right
            if grow_right:
                new_W += pad

            source_bounds = (0, H - 1, 0, W - 1)  # Copy full old map

        # Create new SOM
        new_params = replace(self.satsom.params, grid_shape=(new_H, new_W))
        new_som = SatSOM(new_params).to(self.satsom.weights.device)

        # Transfer
        self._transfer_state(new_som, (offset_r, offset_c), source_bounds)
        self.satsom = new_som
