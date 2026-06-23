import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, mirna_input_dim, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder

        self.mirna_encoder = nn.Sequential(
            nn.Linear(mirna_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

        self.feat_dim = 256

        self.img_proj = nn.Linear(768, 256)

        self.gate = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, self.feat_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, mirna):

        img_feat = self.image_encoder(img)

        img_feat = self.img_proj(img_feat)

        mirna_feat = self.mirna_encoder(mirna)

        fusion_cat = torch.cat([img_feat, mirna_feat], dim=1)
        gate = self.gate(fusion_cat)

        fusion = img_feat + gate * mirna_feat

        logits = self.classifier(fusion)

        return logits
