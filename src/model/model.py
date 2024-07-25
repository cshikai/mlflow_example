from typing import Dict

import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    """
    """

    def __init__(self, parameters: Dict):
        """
        """
        super().__init__()
       
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=parameters['dim_model'],
                                                      nhead=parameters['nhead'],
                                                      dim_feedforward=parameters['dim_feedforward'],
                                                      dropout=parameters['encoder_dropout'],
                                                      activation=parameters['transformer_activation']),
                                                      num_layers=parameters['num_encoder_layers'])
        self.feedfoward = nn.Linear(in_features=parameters['dim_model'],out_features=2)
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(self.transformer_encoder,self.feedfoward,self.softmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
