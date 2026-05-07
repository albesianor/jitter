from torch.nn.functional import cosine_similarity
import torch

class CosScorer:
    """
    A cosine-similarity scorer class.
    """

    def __init__(self, centroid: torch.Tensor):
        """
        Args:
            centroid (torch.Tensor): The centroid for cosine-similarity scoring.
        """
        self.centroid = centroid

    def __call__(self, tensor: torch.Tensor) -> float:
        """
        Args:
            tensor (torch.Tensor): The tensor to score.

        Returns:
            float: The cosine-similarity score.
        """
        return float(cosine_similarity(tensor, self.centroid, dim=1))

    def centroid_size(self) -> torch.Size:
        """
        Returns:
            torch.Size: The size of the centroid tensor.
        """
        return self.centroid.size()
    