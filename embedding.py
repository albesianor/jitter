from transformers import AutoTokenizer, AutoModel
import torch

class HFEmbedder:
    """
    An embedder class using HuggingFace pretrained models.
    """

    def __init__(self, hf_model: str):
        """
        Args:
            hf_model (str): HuggingFace model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModel.from_pretrained(hf_model)

    def __call__(self, sentence: str, method: str = "mean") -> torch.Tensor:
        """
        Embeds a sentence using the selected model.

        Args:
            sentence (str): The sentence to embed.
            method (str): The pooling method: mean (default), CLS.

        Returns:
            torch.Tensor: The mean of the word embeddings.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        
        with torch.inference_mode():
            outputs = self.model(**inputs)

        if method == "CLS":
            return outputs.last_hidden_state[:, 0, :]

        return outputs.last_hidden_state.mean(dim=1)

    def dimension(self) -> int:
        """
        Returns:
            int: The embedding dimension.
        """
        return self.model.config.hidden_size