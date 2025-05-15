from copy import deepcopy
import torch.nn as nn
from pixel_set_encoder import PixelSetEncoder, LinearLayer
from temporal_attn_encoder import TemporalAttentionEncoder

def get_decoder(n_neurons, n_classes):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
        n_classes (int): Output size
    """
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(LinearLayer(n_neurons[i], n_neurons[i + 1]))
    layers.append(nn.Linear(n_neurons[-1], n_classes))
    m = nn.Sequential(*layers)
    return m

class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        n_head=4,
        d_k=32,
        d_model=None,
        mlp3=[512, 128, 128],
        dropout=0.2,
        T=1000,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
        max_position=365,
    ):
        super(PseTae, self).__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += 4
        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = TemporalAttentionEncoder(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_position=max_position,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Positions : Batch_size x Sequence length
           Extra-features : Batch_size x Sequence length x Number of features
        """
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        else:
            return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )
        return total
    

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)