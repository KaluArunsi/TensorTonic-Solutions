import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image):
        self.image = image
        to_float = self.image.float() / 255.0
        permute_image = to_float.permute(2, 0, 1)
        normalised = (permute_image - self.mean) / self.std

        return normalised