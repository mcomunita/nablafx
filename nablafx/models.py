import torch

from typing import List, Union

from nablafx.interfaces import Processor, Controller


class BlackBoxModel(torch.nn.Module):
    """Black box model made of a single neural network processor.
    - processor: processing neural network
    """

    def __init__(
        self,
        processor: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.num_controls = self.processor.num_controls

        print("\nBlackBoxModel:")
        print(self.processor)
        if hasattr(self.processor, "rf"):
            print("Receptive Field: ", self.processor.rf)
        print()

    def forward(self, x: torch.Tensor, controls: torch.Tensor, train: bool = False):
        if self.num_controls > 0:
            return self.processor(x, controls)
        return self.processor(x)

    def reset_states(self):
        if hasattr(self.processor, "reset_states"):
            self.processor.reset_states()

    def detach_states(self):
        if hasattr(self.processor, "detach_states"):
            self.processor.detach_states()


class GreyBoxModel(torch.nn.Module):
    """Class implementing a grey-box model made of a chain of processors and controllers.
    - processor: chain of processing blocks
    - controller: chain of controllers
    """

    def __init__(
        self,
        processors: List[torch.nn.Module],
        num_controls: int = 0,
        stat_control_params_initial: Union[str, float] = 0.0,
        stat_cond_num_layers: int = 3,
        stat_cond_hidden_dim: int = 16,
        dyn_block_size: int = 128,
        dyn_num_layers: int = 1,
        dyn_cond_block_size: int = 128,
        dyn_cond_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_controls = num_controls
        self.processor = Processor(processors)  # init chain of processors
        self.controller = Controller(  # init chain of controllers for the processors
            self.processor,
            num_controls,
            stat_control_params_initial,
            stat_cond_num_layers,
            stat_cond_hidden_dim,
            dyn_block_size,
            dyn_num_layers,
            dyn_cond_block_size,
            dyn_cond_num_layers,
        )

        print("\nGreyBoxModel:")
        print(self.processor)
        print(self.controller)
        print()

    def forward(self, x: torch.Tensor, controls: torch.Tensor = None, train: bool = False):
        """return output and list of dictionaries with the parameters
        for each block in the chain"""
        control_params = self.controller(x, controls)
        y = self.processor(x, control_params, train)
        return y

    def reset_states(self):
        if hasattr(self.processor, "reset_states"):
            self.processor.reset_states()
        if hasattr(self.controller, "reset_states"):
            self.controller.reset_states()

    def detach_states(self):
        if hasattr(self.processor, "detach_states"):
            self.processor.detach_states()
        if hasattr(self.controller, "detach_states"):
            self.controller.detach_states()
