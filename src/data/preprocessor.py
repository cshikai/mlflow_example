
from typing import Dict

import torch
from transformers import AutoTokenizer

class PreProcessor():
    LABEL_MAP = {'negative': 0, 'positive': 1}

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def __call__(self, text: str) -> torch.Tensor:
        text_embedding = self.tokenizer(text,padding='max_length',truncation=True)
        return torch.Tensor(text_embedding['input_ids'])

    def process_label(self, label_string: str) -> int:
        return self.LABEL_MAP[label_string]
    
    def collate(self, batch) -> Dict[str,torch.Tensor]: # list with length of BATCH_SIZE
        keys = batch[0].keys()
        collated_data = {key: [] for key in keys}
        for datapoint in batch:
            for key in datapoint.keys():
                collated_data[key].append(datapoint[key])
        collated_data = {key: torch.stack(value,dim=0) for key,value in collated_data.items()}
        return collated_data
