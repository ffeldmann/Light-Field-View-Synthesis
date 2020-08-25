import copy

import torch
import torch.nn as nn

from LightFieldViewSynthesis.models.keypoint_predictors import ResPoseNet


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim1, dim2):
        super(UnFlatten, self).__init__()
        self.dim1 = dim1
        self.dim1 = dim2

    def forward(self, input):
        return input.reshape(input.size(0), -1, self.dim1, self.dim2)


class AutoEncoder(ResPoseNet):
    def __init__(self, config):
        super(AutoEncoder, self).__init__(config)
        self.config = config
        # If true makes it a Variational AutoEncoder
        self.variational = config["variational"]["active"]
        if self.variational: self.logger.info("Variational AutoEncoder active.")

        if config["load_self_pretrained_encoder"]["active"]:
            path = config["load_self_pretrained_encoder"]["path"]
            self.logger.info(f"Load self pretrained encoder from {path}")
            state_dict = torch.load(path, map_location="cuda")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    name = k.replace("backbone.", "")
                    new_state_dict[name] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)

        if config["load_self_pretrained_decoder"]["active"]:
            path = config["load_self_pretrained_decoder"]["path"]
            self.logger.info(f"Load self pretrained decoder from {path}")
            state_dict = torch.load(path, map_location="cuda")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("head"):
                    name = k.replace("head.", "")
                    new_state_dict[name] = v
            new_state_dict.pop("features.20.weight")
            new_state_dict.pop("features.20.bias")
            self.head.load_state_dict(new_state_dict, strict=False)
        self.head.features[20] = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        # resnet 18 / 34 need different input resnet 50/101/152 : 2048
        if config["resnet_type"] <= 38:
            self.backbone.layer4.add_module("fc", nn.Sequential(
                Flatten(),
                nn.Linear(512 * 4 * 4, config["encoder_latent_dim"])
            ))
            if config["encoder_2"]:
                self.backbone2 = copy.deepcopy(self.backbone)
            if config.get("pose_half", False):
                self.backbone.layer4.add_module("fc", nn.Sequential(
                    Flatten(),
                    nn.Linear(512 * 4 * 4, int(config["encoder_latent_dim"] / 2))
                ))
        else:
            self.backbone.layer4.add_module("fc", nn.Sequential(
                Flatten(),
                nn.Linear(2048 * 4 * 4, config["encoder_latent_dim"])
            ))
            if config["encoder_2"]:
                self.backbone2 = copy.deepcopy(self.backbone)
            if config.get("pose_half", False):
                self.backbone.layer4.add_module("fc", nn.Sequential(
                    Flatten(),
                    nn.Linear(2048 * 4 * 4, int(config["encoder_latent_dim"] / 2))
                ))

        if config["resnet_type"] <= 38:
            # for resnet type 18, 38
            self.fc = nn.Linear(
                config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"], 512 * 4 * 4)
            if config.get("pose_half", False):
                self.fc = nn.Linear(
                    int(config["encoder_latent_dim"] + config["encoder_latent_dim"] / 2), 512 * 4 * 4)
        else:
            # For resnet type 50, 101, 152
            self.fc = nn.Linear(
                config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"], 2048 * 4 * 4)
            if config.get("pose_half", False):
                self.fc = nn.Linear(
                    int(config["encoder_latent_dim"] / 2), 2048 * 4 * 4)
        if self.variational:
            self.fcmu = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])
            self.fcvar = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])
            if config.get("pose_half", False):
                self.fcmu = nn.Linear(int(config["encoder_latent_dim"] / 2), int(config["encoder_latent_dim"] / 2))
                self.fcvar = nn.Linear(int(config["encoder_latent_dim"] / 2), int(config["encoder_latent_dim"] / 2))

        self.tanh = torch.nn.Tanh()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2=None):
        if self.variational:
            x1 = self.backbone(x1)  # [1, 256]
            mu, logvar = self.fcmu(x1), self.fcvar(x1)
            x1 = self.reparameterize(mu, logvar)
            if x2 != None:
                x2 = self.backbone2(x2)
                x1x2 = torch.cat((x1, x2), dim=1)
                # Reshape x1 for Upsampling
                x1x2 = self.fc(x1x2).view(x1x2.size(0), -1, 4, 4)
                return self.head(x1x2), mu, logvar
            # Reshape x1 for Upsampling
            x1 = self.fc(x1).view(x1.size(0), -1, 4, 4)
            x1 = self.head(x1)
            return self.tanh(x1), mu, logvar
        else:
            x1 = self.backbone(x1)
            if x2 != None:
                x2 = self.backbone2(x2)
                x1x2 = torch.cat((x1, x2), dim=1)
                # Reshape x1 for Upsampling
                x1x2 = self.fc(x1x2).view(x1.size(0), -1, 4, 4)
                return self.head(x1x2)
            # Reshape x1 for Upsampling
            x1 = self.fc(x1).view(x1.size(0), -1, 4, 4)
            x1 = self.head(x1)
            return self.tanh(x1)
