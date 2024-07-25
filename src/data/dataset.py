from typing import Dict
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessor import PreProcessor

class MovieSentimentDataset(Dataset):
    """
    """
    FEATURE = 'text'
    LABEL = 'sentiment'
    DATA_ROOT= '/app/data'

    def __init__(self, mode: str, preprocessor: PreProcessor) -> None:
        """
        """
        self.data_path = os.path.join(self.DATA_ROOT, mode)
        self.data = pd.read_csv(os.path.join(self.data_path,'movie_sentiment_{}_data.csv'.format(mode)))
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        '''
        Get the length of dataset.
        '''
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str,torch.Tensor]:
        data_slice = self.data.iloc[index]
        x = data_slice[self.FEATURE]
        x = self.preprocessor(x)
        y = data_slice[self.LABEL]
        y = self.preprocessor.process_label(y)
        y = torch.from_numpy(np.array([y]))
        return {'text' : x, 'label': y}
