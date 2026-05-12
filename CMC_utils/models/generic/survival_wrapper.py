import torch
from typing import Union
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate


__all__ = ["SurvivalWrapper"]


class SurvivalWrapper(torch.nn.Module):
    """
    Wrapper for survival models. It takes a shared network and a list of subnetworks, one for each event.
    The shared network is used to extract features from the input data, and the subnetworks are used to predict the
    probability of each event at each time step.
    """
    def __init__(self, num_events: int, max_time: int, shared_net: Union[dict, DictConfig], cs_subnet: Union[dict, DictConfig], **kwargs):
        super(SurvivalWrapper, self).__init__()

        self.shared_net = instantiate(shared_net["init_params"])

        self.input_size = self.shared_net.input_size
        self.output_size = num_events * max_time
        self.num_events = num_events
        self.max_time = max_time

        cs_subnet["init_params"]["input_size"] = self.shared_net.output_size
        cs_subnet["init_params"]["output_size"] = max_time

        self.cs_subnets = torch.nn.ModuleList()
        for k in range(num_events):
            subnet = instantiate(cs_subnet["init_params"])
            self.cs_subnets.append(subnet)

    def forward(self, inputs):

        x = self.shared_net(inputs)

        y = []
        for subnet in self.cs_subnets:
            x_CS = subnet(x)
            y_CS = F.softmax(x_CS, dim=-1)
            y.append(y_CS)
        y = torch.cat(y, dim=-1)

        return y


if __name__ == "__main__":
    pass
