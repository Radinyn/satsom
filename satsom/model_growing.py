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
            return active_2d, (0, H - 1, 0, W - 1)

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
        """
        GPU-safe transfer of a rectangular block of neurons from self.satsom -> new_som.
        """
        r_min, r_max, c_min, c_max = old_bounds
        r_off, c_off = offset

        H_old, W_old = self.satsom.params.grid_shape
        H_new, W_new = new_som.params.grid_shape

        num_neurons = self.satsom.num_neurons
        D = self.satsom.params.input_dim
        C = self.satsom.params.output_dim

        # Quick sanity on bounds
        assert 0 <= r_min <= r_max < H_old, f"Row bounds out of range: {r_min},{r_max}"
        assert 0 <= c_min <= c_max < W_old, f"Col bounds out of range: {c_min},{c_max}"

        h_slice = r_max - r_min + 1
        w_slice = c_max - c_min + 1
        slice_len = h_slice * w_slice

        # Ensure all parameter tensors are on the same device and have expected dtypes
        dev = self.satsom.weights.device
        # align new_som device as well
        assert new_som.weights.device == dev, (
            "new_som tensors must live on same device as self.satsom"
        )

        # ensure tensors are float where expected
        assert self.satsom.weights.dtype.is_floating_point, "weights must be float"
        assert new_som.weights.dtype.is_floating_point, "new weights must be float"

        # Build old (source) flat indices on correct device / dtype
        row_indices_old = torch.arange(r_min, r_max + 1, device=dev, dtype=torch.long)
        col_indices_old = torch.arange(c_min, c_max + 1, device=dev, dtype=torch.long)

        r_old_idx, c_old_idx = torch.meshgrid(
            row_indices_old, col_indices_old, indexing="ij"
        )
        flat_old_idx = (r_old_idx * W_old + c_old_idx).reshape(
            -1
        )  # shape: (slice_len,)

        # Sanity checks: indices within [0, num_neurons-1]
        if flat_old_idx.numel() > 0:
            max_idx = int(flat_old_idx.max().item())
            min_idx = int(flat_old_idx.min().item())
            assert min_idx >= 0, f"Computed flat_old_idx has negative entry: {min_idx}"
            assert max_idx < num_neurons, (
                f"Computed flat_old_idx out of range: {max_idx} >= {num_neurons}"
            )

        # Extract old slices using flat indexing; use index_select for clarity
        with torch.no_grad():
            # shape (slice_len, D)
            old_flat_w = self.satsom.weights.index_select(0, flat_old_idx)
            old_flat_lr = self.satsom.learning_rates.index_select(0, flat_old_idx)
            old_flat_sig = self.satsom.sigmas.index_select(0, flat_old_idx)
            old_flat_lbl = self.satsom.labels.index_select(0, flat_old_idx)

            # reshape to 2D grid form (h_slice, w_slice, ...)
            old_slice_w = old_flat_w.view(h_slice, w_slice, D)
            old_slice_lr = old_flat_lr.view(h_slice, w_slice)
            old_slice_sig = old_flat_sig.view(h_slice, w_slice)
            old_slice_lbl = old_flat_lbl.view(h_slice, w_slice, C)

            # Build new (target) wrapped indices on same device
            row_indices_new = (
                torch.arange(r_off, r_off + h_slice, device=dev, dtype=torch.long)
                % H_new
            )
            col_indices_new = (
                torch.arange(c_off, c_off + w_slice, device=dev, dtype=torch.long)
                % W_new
            )

            r_new_idx, c_new_idx = torch.meshgrid(
                row_indices_new, col_indices_new, indexing="ij"
            )
            flat_new_idx = (r_new_idx * W_new + c_new_idx).reshape(-1)

            # Sanity checks for new indices
            if flat_new_idx.numel() > 0:
                max_new = int(flat_new_idx.max().item())
                min_new = int(flat_new_idx.min().item())
                assert min_new >= 0 and max_new < new_som.num_neurons, (
                    f"Computed flat_new_idx out of range: [{min_new}, {max_new}] vs num_neurons {new_som.num_neurons}"
                )

            # Finally scatter into new_som's flat parameter tensors
            # reshape old slices into (slice_len, ...)
            new_flat_w = old_slice_w.reshape(slice_len, D)
            new_flat_lr = old_slice_lr.reshape(slice_len)
            new_flat_sig = old_slice_sig.reshape(slice_len)
            new_flat_lbl = old_slice_lbl.reshape(slice_len, C)

            # assign in-place
            new_som.weights[flat_new_idx] = new_flat_w
            new_som.learning_rates[flat_new_idx] = new_flat_lr
            new_som.sigmas[flat_new_idx] = new_flat_sig
            new_som.labels[flat_new_idx] = new_flat_lbl

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
