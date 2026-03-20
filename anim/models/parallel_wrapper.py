"""
Thin wrapper that provides a uniform interface regardless of whether the model
is wrapped in nn.DataParallel or running on a single device.
"""
# External
import torch


class ParallelWrapper:
    """
    Delegates all calls to the underlying model while transparently handling
    the nn.DataParallel ``module`` indirection.

    Attributes:
        model:  The (possibly DataParallel-wrapped) nn.Module.
        mmodel: The unwrapped underlying model (i.e. model.module if DataParallel).
    """

    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.mmodel = model.module if hasattr(model, 'module') else model

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device, dtype):
        self.model = self.model.to(device, dtype)
        return self

    # ------------------------------------------------------------------
    # Model interface delegation
    # ------------------------------------------------------------------

    @property
    def mode3d(self):
        return self.mmodel.mode3d

    def reset(self):
        self.mmodel.reset()

    def forward_pass(self, model_input, model_target, optimise: bool = False) -> dict:
        return self.mmodel.forward_pass(model_input, model_target, optimise=optimise)

    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        self.mmodel.epoch_end(epoch, train_losses, val_losses)

    # ------------------------------------------------------------------
    # Parameter / state-dict access
    # ------------------------------------------------------------------

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    @property
    def optim(self):
        return self.mmodel.optim

    @property
    def lr_scheduler(self):
        return self.mmodel.lr_scheduler
