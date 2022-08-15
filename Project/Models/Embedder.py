import torch.nn as nn


class AlexNetSE(nn.Module):
    """
    AlexNet for spatial embedding
    """

    def __init__(self, drop_p=0, emb_dim=4096, bn_flag=False):
        super(AlexNetSE, self).__init__()
        self.batch_norm_layer = nn.BatchNorm2d(1) if bn_flag else None
        self.layers = []
        self.layers.append(nn.Sequential(
            #  TODO: not sure that in_channels should be 3, the embedding is for the videos itself too.
            #   Also, note it expect batch_size to be first
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # TODO: padding should be int (was padding="valid")
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding="valid"),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding="valid"),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding="valid"),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ))

        self.layers.append(nn.Sequential(
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(drop_p)
        ))
        self.layers.append(nn.Sequential(
            nn.Linear(in_features=4096, out_features=emb_dim),
            nn.ReLU(),
            nn.Dropout(drop_p)
        ))

    def forward(self, X):
        out = X if self.batch_norm_layer is None else self.batch_norm_layer(X)
        out = out.permute(1, 2, 0, 3)
        for i in range(len(self.layers)):
            if i == 5:
                out = out.view(-1, 6 * 6 * 256)  # TODO: not sure what are those numbers, can you put them in contants/add comment?
            out = self.layers[i](out)
        return out
