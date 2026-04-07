import torch
import torch.nn as nn


class ToneCurve(nn.Module):
    """Per-image polynomial tone mapping for low-light 3DGS training.

    For each training image i and channel c, applies:
        T_{i,c}(x) = a0 + a1*x + a2*x^2 + ... + a_D*x^D

    Initialized to identity (a1=1, rest=0) so T(x)=x at start.
    """

    def __init__(self, num_images, degree=3):
        super().__init__()
        self.degree = degree
        # coeffs: (num_images, 3, degree+1)
        coeffs = torch.zeros(num_images, 3, degree + 1)
        coeffs[:, :, 1] = 1.0  # a1 = 1 -> identity mapping
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, rendered, image_idx):
        """
        Args:
            rendered: (H, W, 3) rendered image in [0, 1]
            image_idx: int, index of the training image
        Returns:
            mapped: (H, W, 3) tone-mapped image, clamped to [0, 1]
        """
        c = self.coeffs[image_idx]  # (3, degree+1)
        x = rendered  # (H, W, 3)

        # Polynomial evaluation: a0 + a1*x + a2*x^2 + ...
        result = c[:, 0]  # (3,) broadcast to (H, W, 3)
        x_pow = x
        for d in range(1, self.degree + 1):
            result = result + c[:, d] * x_pow
            if d < self.degree:
                x_pow = x_pow * x

        return torch.clamp(result, 0.0, 1.0)
