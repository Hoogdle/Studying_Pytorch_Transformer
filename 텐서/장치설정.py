import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
cpu = torch.FloatTensor([1,2,3])
gpu = torch.cuda.FloatTensor([1,2,3])
tensor = torch.rand((1,1),device=device)
print(device)
print(cpu)
print(gpu)
print(tensor)

