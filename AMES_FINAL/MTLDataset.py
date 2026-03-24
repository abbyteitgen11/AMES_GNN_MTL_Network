from torch.utils.data import Dataset

class MTLDataset(Dataset):
    def __init__(self, features, targets):
        """
            A class that converts tensors to dataset for use with pytorch

            Args:
            :param features: input data
            :param targets: target data
        """

        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return feature and target pair for each example
        return self.features[idx], self.targets[idx]