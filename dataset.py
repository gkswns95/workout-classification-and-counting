from torch.utils.data import Dataset

class CustomDataset(Dataset):
    r"""custom pytorch datase
    input data with 2 item(class label, count value)
    Arguments: 
    x (:obj:`list`):
    data item list. 
    y (:obj:`list[tuple]`)
    list of tuple. each item with 2 item(class label, count value)
    """
    def __init__(self,x,y):
        self.x_data = x
        self.y_data = y

    def __len__(self):
        r"""When used `len` return the number of examples.
        """
        return len(self.x_data)

    def __getitem__(self, idx):
        r"""Given an index return an example from the position.
        Arguments:
          item (:obj:`int`):
              Index position to pick an example to return.
        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
      
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y