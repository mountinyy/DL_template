import pandas as pd
from torch.utils.data import Dataset


class GoEmotionsDataset(Dataset):
    def __init__(self, path: str, usage: str):
        """

        Args:
            path: dataset path for refined GoEmotions
            usage: train / valid / test
        """
        assert usage in ["train", "valid", "test"], "지원하지 않는 usage입니다."
        df = pd.read_csv(path)
        self.context = df["text"].values
        self.labels = df["emotion"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """

        Args:
            idx (int): data idx

        Returns:
            dict{'context': text, 'labels': emotion label}
        """
        return {"context": self.context[idx], "label": self.labels[idx]}
