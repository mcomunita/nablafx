import torch

from typing import List

from nablafx.controllers import DummyController, StaticController, StaticCondController, DynamicController, DynamicCondController


class Processor(torch.nn.Module):
    """Processor class is made of a chain of processors.
    Each processor receives a set of control parameters from a controller.
    """

    def __init__(
        self,
        processors: List[torch.nn.Module],
    ):
        super().__init__()
        self.processors = torch.nn.ModuleList(processors)
        self.num_control_params = [p.num_control_params for p in self.processors]
        self.tot_control_params = sum(self.num_control_params)

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        # params = [] # list of dicts of params for each processor, used for logging
        for prc, ctrl_prms in zip(self.processors, control_params):
            x, _ = prc(x, ctrl_prms, train=train)
            # params.append(param_dict)
        return x

    def reset_states(self):
        for prc in self.processors:
            if hasattr(prc, "reset_states"):
                prc.reset_states()


class Controller(torch.nn.Module):
    """Controller class is made of a chain of controllers.
    Each controller (optionally) receives an input signal and
    a set of control values from a dataset and returns
    a set of control parameters for a processor.
    """

    def __init__(
        self,
        processor: Processor,
        num_controls: int,
        stat_control_params_initial: float = 0.0,
        stat_cond_num_layers: int = 3,
        stat_cond_hidden_dim: int = 16,
        dyn_block_size: int = 128,
        dyn_num_layers: int = 1,
        dyn_cond_block_size: int = 128,
        dyn_cond_num_layers: int = 1,
    ):
        super().__init__()
        self.num_controls = num_controls

        self.controllers = []
        for prc in processor.processors:
            if prc.control_type is None:
                self.controllers.append(DummyController())
            elif prc.control_type == "static":
                self.controllers.append(
                    StaticController(
                        prc.num_control_params,
                        stat_control_params_initial,
                        prc.lr_multiplier,
                    )
                )
            elif prc.control_type == "static-cond":
                self.controllers.append(
                    StaticCondController(
                        num_controls,
                        prc.num_control_params,
                        stat_cond_num_layers,
                        stat_cond_hidden_dim,
                        prc.lr_multiplier,
                    )
                )
            elif prc.control_type == "dynamic":
                self.controllers.append(
                    DynamicController(
                        prc.num_control_params,
                        dyn_block_size,
                        dyn_num_layers,
                        prc.lr_multiplier,
                    )
                )
            elif prc.control_type == "dynamic-cond":
                self.controllers.append(
                    DynamicCondController(
                        num_controls,
                        prc.num_control_params,
                        dyn_cond_block_size,
                        dyn_cond_num_layers,
                        prc.lr_multiplier,
                    )
                )
            else:
                raise ValueError(f"Unknown control type {type(prc.control_type)}")
        self.controllers = torch.nn.ModuleList(self.controllers)

    def forward(self, x: torch.Tensor, controls: torch.Tensor = None):
        if self.num_controls > 0:
            assert x.shape[0] == controls.shape[0]  # batch size
            assert controls.shape[1] == self.num_controls  # num controls
        control_params = []
        for ctrl in self.controllers:
            if isinstance(ctrl, DummyController):
                control_params.append(ctrl(x=x))
            if isinstance(ctrl, StaticController):
                control_params.append(ctrl(x=x))
            elif isinstance(ctrl, StaticCondController):
                control_params.append(ctrl(controls=controls))
            elif isinstance(ctrl, DynamicController):
                control_params.append(ctrl(x=x))
            elif isinstance(ctrl, DynamicCondController):
                control_params.append(ctrl(x=x, controls=controls))
        return control_params

    def reset_states(self):
        for ctrl in self.controllers:
            if hasattr(ctrl, "reset_states"):
                ctrl.reset_states()

    def detach_states(self):
        for ctrl in self.controllers:
            if hasattr(ctrl, "detach_states"):
                ctrl.detach_states()
