import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class CNNBlock(nn.Module):

    out_put_size : dict
    input_height: int
    input_width: int
    pool_depth: int
    phases: int
    last_phase: int
    layers: nn.ModuleList
    pool: nn.MaxPool2d
    batch_norm: nn.BatchNorm2d

    def __init__(self, 
                 feature: list = [1,16,16,32,32,64,64,64], 
                 height: int = 134, 
                 width: int = 134, 
                 pool_depth: int = 3,
                 conv_kernel_size: int = 3,
                 conv_padding: int = 1,
                 pool_kernel_size: int = 2,
                 pool_kernel_stride: int = 2
                 ):
        super(CNNBlock, self).__init__()

        # Check if depth pool remains in the domain
        if pool_depth > len(feature):
            warnings.warn(f"The pool depth has been set to {len(feature)} since it cannot be more. Your biological tree is a circle", UserWarning)

        # Assign default values
        self.input_height = height
        self.input_width = width
        self.pool_depth = pool_depth
        self.phases = len(feature) // self.pool_depth
        self.last_phase = len(feature) % self.pool_depth
        self.out_put_size = {}

        # Build the pool layer
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_stride)

        # Build the convs layers
        self.layers = nn.ModuleList([
            nn.Conv2d(feature[i], feature[i + 1], kernel_size=conv_kernel_size, padding=conv_padding)
            for i in range(len(feature) - 1)
        ])
        
        # Build the norm layers
        self.batch_norm = nn.BatchNorm2d(feature[len(feature)-1])

        # Calculate the out
        self.out_put_size["features"] = feature[len(feature)-1]

        self.out_put_size["height"] = self.input_height
        self.out_put_size["width"] = self.input_width
        
        for _ in range(self.phases + int(self.last_phase != 0 )):
            self.out_put_size["height"] = int( ((self.out_put_size["height"] - pool_kernel_size)/pool_kernel_stride)+1 )
            self.out_put_size["width"] = int( ((self.out_put_size["width"] - pool_kernel_size)/pool_kernel_stride)+1 )

        # Raise a warning the the output makes no sense
        if (self.out_put_size["height"] <= 1) or (self.out_put_size["width"] <= 1):
            warnings.warn(f"The output features are less or equal than 1x1, this makes no sense you fucking moron", UserWarning)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        # Check if the input size is correct according to the design
        if x[0].size() != (1, self.input_height, self.input_width):
            warnings.warn(f"The input size should be (batch, 1, {self.input_height}, {self.input_width}), got {x.size()} instead. Fix it you dick", UserWarning)

        # Conv and pool steps
        iterator = 0
        for _ in range(self.phases):
            for _ in range(self.pool_depth):
                x =  F.relu(self.layers[iterator](x))
                iterator += 1
            x = self.pool(x)
        
        for _ in range(self.last_phase):
            x =  F.relu(self.layers[iterator](x))
        x = self.pool(x)

        # normalize and return the result
        return self.batch_norm(x)
    
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    height=600
    width=600

    model = CNNBlock(height=height, width=width)

    input_tensor = torch.randn(32,1, 600, 600)  

    output = model(input_tensor)

    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {model.n_parameters()}")
    print(model.out_put_size)