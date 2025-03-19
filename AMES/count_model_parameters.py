import torch

def count_model_parameters(model: torch.nn.Module) -> int:

    """ 
    A function to count model parameters

    :param: model for which parameters are going to be counted
    :type: torch.nn.Module

    :rtype: int

    """

    n_parameters = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            n_p = 1
            for m in range(len(parameter.shape)):

                n_p *= parameter.shape[m]

            n_parameters += n_p

    return n_parameters
