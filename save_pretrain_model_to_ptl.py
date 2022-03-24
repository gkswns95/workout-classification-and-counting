import torch

from model import *
from torch.utils.mobile_optimizer import optimize_for_mobile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN2D(p=0).to(device)
model.eval()

example = torch.load('./example/x_example.pt')

traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter('test_ptl.ptl')