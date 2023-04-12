# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.models import FretNet

from amt_tools.models import OnlineLanguageModel

# Regular imports
import torch


class FretNetRecurrent(FretNet):
    """
    Implements FretNet with a recurrent layer inserted at the front of the tablature head.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model and insert the recurrent layer.

        Parameters
        ----------
        See FretNet class...
        """

        super().__init__(**kwargs)

        # Break off the first linear layer in the tablature head so it can be replaced
        linear_layer, tablature_head = self.tablature_head[0], self.tablature_head[1:]

        # Insert the recurrent layer before the output layer
        self.tablature_head = torch.nn.Sequential(
            OnlineLanguageModel(dim_in=linear_layer.in_features,
                                dim_out=linear_layer.out_features),
            tablature_head
        )
