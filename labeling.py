from transformers import pipeline

class HFGoodBadLabeler:
    """
    A zero-shot labeler using pretrained HuggingFace models and two label categories.
    """

    def __init__(
        self,
        classifier: pipeline,
        good: list,
        bad: list,
        high: float = 0.5,
        low: float = 0.5,
        hard_cutoff: float = 0.5,
    ):
        """
        Args:
            classifier (transformers.pipeline): HuggingFace classifier pipeline.
            good (list(str)): List of labels in the "good" category.
            bad (list(str)): List of labels in the "bad" category.
            high (float): Threshold for "good" labels.
            low (float): Threshold for "bad" labels.
            hard_cutoff (float): Threshold for highest category to be determinant.
        """

        self.classifier = classifier
        self.classes = good + bad
        self.good = set(good)
        self.high = high
        self.low = low
        self.hard_cutoff = hard_cutoff

    def __call__(self, sequence: str) -> int:
        """
        Labels according to good and bad categories.

        Args:
            sequence (str): Sequence to label.

        Returns:
            int: 0 if sum(good) < self.low OR any bad > self.hard_cutoff
                 1 if sum(good) > self.high OR any good > self.hard_cutoff
                 -1 otherwise
        """

        classification = self.classifier(sequence, self.classes)

        # test if highest category is in good or bad
        if classification["scores"][0] > self.hard_cutoff:
            label = classification["labels"][0]
            
            if label in self.good:
                return 1
            else:
                return 0

        # compute good score
        good_score = 0
        for i, label in enumerate(classification["labels"]):
            if label in self.good:
                good_score += classification["scores"][i]

        if good_score > self.high:
            return 1
        
        if good_score < self.low:
            return 0

        return -1