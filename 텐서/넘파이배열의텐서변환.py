import torch
import numpy as np

ndarray = np.array([1,2,3],dtype=np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))