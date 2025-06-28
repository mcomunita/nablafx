import torch
import auraloss
from torch import Tensor as T
from auraloss.utils import apply_reduction


class PreEmph(torch.nn.Module):
    """Pre-emphasis filter module."""

    def __init__(self, filter_cfs, low_pass=0):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = torch.nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = torch.nn.Conv1d(1, 1, 2, bias=False)
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
        target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)


class EDCLoss(torch.nn.Module):
    """Energy Decay Curve loss function module."""

    def __init__(self, remove_dc=True, eps=1e-8, clamp=True, min_db=-80, max_db=0, error_fcn=torch.nn.MSELoss(), reduction="mean"):
        super().__init__()
        self.remove_dc = remove_dc
        self.eps = eps
        self.clamp = clamp
        self.min_db = min_db
        self.max_db = max_db
        self.error_fcn = error_fcn
        self.reduction = reduction

    def forward(self, input: T, target: T) -> T:
        if self.remove_dc:
            input = input - torch.mean(input, dim=-1, keepdim=True)
            target = target - torch.mean(target, dim=-1, keepdim=True)

        if self.clamp:
            input = torch.abs(input)
            input = 20.0 * torch.log10(input + self.eps)
            input = torch.clamp(input, self.min_db, self.max_db)
            input = torch.pow(10.0, input / 20.0)

            target = torch.abs(target)
            target = 20.0 * torch.log10(target + self.eps)
            target = torch.clamp(target, self.min_db, self.max_db)
            target = torch.pow(10.0, target / 20.0)

        # Schroeder's energy decay curve
        input_energy = 10.0 * torch.log10(torch.sum(torch.square(input)) + self.eps)
        target_energy = 10.0 * torch.log10(torch.sum(torch.square(target)) + self.eps)

        input_edc = (
            10.0 * torch.log10(torch.flip(torch.cumsum(torch.flip(torch.square(input), dims=[-1]), dim=-1), dims=[-1])) - input_energy
        )
        target_edc = (
            10.0 * torch.log10(torch.flip(torch.cumsum(torch.flip(torch.square(target), dims=[-1]), dim=-1), dims=[-1])) - target_energy
        )

        losses = self.error_fcn(input_edc, target_edc)
        losses = apply_reduction(losses, reduction=self.reduction)

        return losses


class LossWrapper(torch.nn.Module):
    def __init__(self, losses, pre_filt=None):
        super(LossWrapper, self).__init__()
        self.losses = losses
        self.loss_dict = {
            "ESR": auraloss.time.ESRLoss(),
            "DC": auraloss.time.DCLoss(),
            "L1": torch.nn.L1Loss(),
            "STFT": auraloss.freq.STFTLoss(),
            "MSTFT": auraloss.freq.MultiResolutionSTFTLoss(),
            "EDC": EDCLoss(),
        }
        if pre_filt:
            print("prefilt")
            pre_filt = PreEmph(pre_filt)
            self.loss_dict["ESRPre"] = lambda output, target: self.loss_dict["ESR"].forward(*pre_filt(output, target))
        loss_functions = [[self.loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        all_losses = {}
        for i, loss in enumerate(self.losses):
            # original shape: length x batch x 1
            # auraloss needs: batch x 1 x length
            loss_fcn = self.loss_functions[i]
            loss_factor = self.loss_factors[i]
            # if isinstance(loss_fcn, auraloss.freq.STFTLoss) or isinstance(loss_fcn, auraloss.freq.MultiResolutionSTFTLoss):
            #     output = torch.permute(output, (1, 2, 0))
            #     target = torch.permute(target, (1, 2, 0))
            all_losses[loss] = torch.mul(loss_fcn(output, target), loss_factor)
        return all_losses


class ESRandDCLoss(torch.nn.Module):
    def __init__(
        self,
        esr_loss: torch.nn.Module,
        dc_loss: torch.nn.Module,
        esr_weight: float = 5.0,
        dc_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.esr_loss = esr_loss
        self.dc_loss = dc_loss
        self.esr_weight = esr_weight
        self.dc_weight = dc_weight

    def forward(self, x, y):
        esr_loss = self.esr_loss(x, y)
        dc_loss = self.dc_loss(x, y)
        return (self.esr_weight * esr_loss, self.dc_weight * dc_loss)


class TimeAndFrequencyDomainLoss(torch.nn.Module):
    def __init__(
        self,
        time_domain_loss: torch.nn.Module,
        frequency_domain_loss: torch.nn.Module,
        time_domain_weight: float = 5.0,
        frequency_domain_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.time_domain_loss = time_domain_loss
        self.frequency_domain_loss = frequency_domain_loss
        self.time_domain_weight = time_domain_weight
        self.frequency_domain_weight = frequency_domain_weight

    def forward(self, x, y):
        td_loss = self.time_domain_loss(x, y)
        fd_loss = self.frequency_domain_loss(x, y)
        return (self.time_domain_weight * td_loss, self.frequency_domain_weight * fd_loss)


class WeightedMultiLoss(torch.nn.Module):
    """
    A flexible loss function that combines multiple loss functions with individual weights.
    Each loss function can have its own parameters and weight.
    
    Args:
        losses: List of dictionaries, each containing:
            - 'loss': The loss function instance
            - 'weight': The weight for this loss function
            - 'name': Optional name for logging purposes
    
    Example:
        losses = [
            {
                'loss': auraloss.time.ESRLoss(),
                'weight': 0.7,
                'name': 'esr'
            },
            {
                'loss': auraloss.freq.MultiResolutionSTFTLoss(),
                'weight': 0.3,
                'name': 'multi_stft'
            }
        ]
        multi_loss = WeightedMultiLoss(losses)
    """
    
    def __init__(self, losses: list) -> None:
        super().__init__()
        
        if not losses:
            raise ValueError("At least one loss function must be provided")
        
        self.loss_functions = torch.nn.ModuleList()
        self.weights = []
        self.names = []
        
        for i, loss_config in enumerate(losses):
            if not isinstance(loss_config, dict):
                raise ValueError(f"Loss config {i} must be a dictionary")
            
            if 'loss' not in loss_config:
                raise ValueError(f"Loss config {i} must contain 'loss' key")
            
            if 'weight' not in loss_config:
                raise ValueError(f"Loss config {i} must contain 'weight' key")
            
            loss_fn = loss_config['loss']
            weight = loss_config['weight']
            name = loss_config.get('name', f'loss_{i}')
            
            if not isinstance(loss_fn, torch.nn.Module):
                raise ValueError(f"Loss function {i} must be a torch.nn.Module")
            
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for loss {i} must be a number")
            
            self.loss_functions.append(loss_fn)
            self.weights.append(float(weight))
            self.names.append(name)
    
    def forward(self, x, y):
        """
        Compute weighted sum of all loss functions.
        
        Returns:
            If only one loss function: returns the weighted loss value
            If multiple loss functions: returns tuple of (individual_losses..., total_loss)
        """
        individual_losses = []
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            loss_value = loss_fn(x, y)
            weighted_loss = weight * loss_value
            individual_losses.append(weighted_loss)
            total_loss += weighted_loss
        
        # Return format consistent with existing TimeAndFrequencyDomainLoss
        if len(individual_losses) == 1:
            return individual_losses[0]
        else:
            return tuple(individual_losses + [total_loss])
    
    def get_loss_names(self):
        """Return list of loss function names for logging purposes."""
        return self.names
    
    def get_weights(self):
        """Return list of weights for each loss function."""
        return self.weights
