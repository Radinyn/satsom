import torch
from satsom import SatSOM, SatSOMParameters

# 1. Define parameters
params = SatSOMParameters(
    grid_shape=(10, 10),  # 10×10 grid of neurons
    input_dim=128,  # 128-dimensional input features
    output_dim=10,  # 10 classes (labels)
    initial_lr=0.1,  # initial learning rate
    initial_sigma=3.0,  # initial neighborhood radius
    Lr=0.01,  # learning rate decay factor
    Lr_bias=0.1,  # neighborhood bias multiplier
    Lr_sigma=0.005,  # sigma decay factor
)

# 2. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SatSOM(params).to(device)

# 3. Single training step
x = torch.randn(params.input_dim).to(device)
y = (
    torch.nn.functional.one_hot(torch.tensor(3), num_classes=params.output_dim)
    .float()
    .to(device)
)
model.train()
model.step(x, y)

# 4. Inference
model.eval()
data = torch.randn(32, params.input_dim).to(device)
out = model(data)  # (batch_size, output_dim)
predictions = out.argmax(dim=1)
