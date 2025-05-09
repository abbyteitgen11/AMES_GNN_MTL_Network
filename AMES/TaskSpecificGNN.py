import torch.nn as nn

class TaskSpecificGNN(nn.Module):
    def __init__(self, model, task_idx, model_args):
        super(TaskSpecificGNN, self).__init__()
        self.model = model
        self.task_idx = task_idx
        self.model_args = model_args

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        outputs = self.model(x, edge_index, edge_attr, batch, *self.model_args)
        task_output = outputs[self.task_idx]
        return task_output.squeeze(-1)