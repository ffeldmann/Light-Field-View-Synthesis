import torch
import torch.nn.functional
import torch.optim as optim
import wandb
from edflow import TemplateIterator
from edflow.util import retrieve
import numpy as np

class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = torch.nn.MSELoss()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # if retrieve(self.config, "integrations/wandb/active", default=False):
        #     wandb.watch(self.model)

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


    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        self.model = self.model.to(self.device)
        self.model.train(is_train)

        # (batch_size, width, height, channel)
        # import pdb;pdb.set_trace()
        hor, vert = kwargs['horizontal'], kwargs['vertical']
        hor = to_torch(hor).to(self.device).permute(0,4,1,2,3)
        vert = to_torch(vert).to(self.device).permute(0,4,1,2,3)
        # compute model
        predictions = self.model(hor, vert)
        # compute loss
        # if retrieve(self.config, "LossConstrained/active", default=False):
        #     predictions, mu, logvar = predictions
        #     loss, log, loss_train_op = self.loss_constrained(inputs,
        #                                                      predictions,
        #                                                      mu,
        #                                                      logvar,
        #                                                      self.get_global_step())
        #     if is_train:
        #         loss_train_op()
        # else:
        #     losses = self.criterion(kwargs["targets"], predictions)
        loss_h = self.criterion(predictions[0], hor)
        loss_v = self.criterion(predictions[1], vert)
        loss = loss_h + loss_v

        # if retrieve(self.config, "variational/active", default=False):
            # Split the predictions tensor in predictions, mu and logvar
            # predictions, mu, logvar = predictions

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():

            logs = {
                "images": {
                    "v_input": vert,
                    "h_input": hor,
                    "h_output": predictions[0],
                    "v_output": predictions[1]
                },
                "scalars": {
                    "loss_h": loss_h,
                    "loss_v": loss_v,
                    "loss": loss
                },
            }

            return logs


        def eval_op():
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

def to_numpy(x, permute=False):
    """automatically detach and move to cpu if necessary."""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
    if isinstance(x, np.ndarray):
        if permute:
            x = np.transpose(x, (0, 2, 3, 1))  # NCHW --> NHWC
    return x

def to_torch(x, permute=False):
    """automatically convert numpy array to torch and permute channels from NHWC to NCHW"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = x.to(device)

    if permute:
        x = x.permute((0, 3, 1, 2))  # NHWC --> NCHW
    if x.dtype is torch.float64:
        x = x.type(torch.float32)
    return x
