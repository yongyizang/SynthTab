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

        # Break apart the tablature head
        linear_layer_t = self.tablature_head[0]
        head_contents_t = self.tablature_head[1 : -1]
        tablature_layer_t = self.tablature_head[-1]

        # Reconstruct the tablature head with a recurrent layer
        self.tablature_head = torch.nn.Sequential(
            linear_layer_t,
            head_contents_t,
            OnlineLanguageModel(dim_in=linear_layer_t.out_features,
                                dim_out=tablature_layer_t.dim_in),
            tablature_layer_t
        )

        if self.estimate_onsets:
            # Break apart the onset detection head
            linear_layer_o = self.onsets_head[0]
            head_contents_o = self.onsets_head[1: -1]
            tablature_layer_o = self.onsets_head[-1]

            # Insert the recurrent layer in place of the linear layer
            self.onsets_head = torch.nn.Sequential(
                linear_layer_o,
                head_contents_o,
                OnlineLanguageModel(dim_in=linear_layer_o.out_features,
                                    dim_out=tablature_layer_o.dim_in),
                tablature_layer_o
            )
