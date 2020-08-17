import torch
import torch.nn.functional
import torch.optim as optim
import wandb
from edflow import TemplateIterator
from edflow.util import retrieve

from LightFieldViewSynthesis.utils.LossConstrained import LossConstrained
from LightFieldViewSynthesis.utils.perceptual_loss.models import PerceptualLoss
from LightFieldViewSynthesis.utils.tensor_utils import sure_to_numpy, sure_to_torch


# from edflow.data.util import adjust_support


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False

        if self.cuda:
            self.model.cuda()

        if retrieve(self.config, "integrations/wandb/active", default=False):
            wandb.watch(self.model)

        # initalize perceptual loss if necessary
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)

        self.loss_constrained = LossConstrained(self.config)

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def criterion(self, targets, predictions):
        # make sure everything is a torch tensor
        targets = sure_to_numpy(targets)
        predictions = sure_to_numpy(predictions)

        batch_losses = {}
        if self.config["losses"]["L2"]:
            batch_losses["L2"] = self.mse_loss(targets, predictions.cpu())

        if self.config["losses"]["perceptual"]:
            batch_losses["perceptual"] = torch.mean(
                self.perceptual_loss(torch.from_numpy(targets),
                                     torch.from_numpy(predictions),
                                     True)).cpu()
        batch_losses["total"] = sum(
            [
                batch_losses[key]
                for key in batch_losses.keys()
            ]
        )
        return batch_losses

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # (batch_size, width, height, channel)
        inputs = sure_to_torch(kwargs["inp"].transpose(0, 3, 1, 2)).to(self.device)

        # compute model
        predictions = model(inputs)
        # compute loss
        losses = self.criterion(kwargs["targets"], predictions.cpu())

        def train_op():
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()

        def log_op():
            logs = {
                "images": {
                    "image_input": sure_to_numpy(inputs).transpose(0, 2, 3, 1),
                    "outputs": sure_to_numpy(predictions).transpose(0, 2, 3, 1),
                    "targets": kwargs["targets"].transpose(0, 2, 3, 1),
                },
                "scalars": {
                    "loss": losses["batch"]["total"],
                },
            }

            return logs

        def eval_op():
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
