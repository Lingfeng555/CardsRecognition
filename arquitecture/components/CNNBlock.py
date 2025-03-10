import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class CNNBlock(nn.Module):

    out_put_size : dict

    def __init__(self, 
                 feature: list = [1,16,16,32,32,64,64,64], 
                 height: int = 134, 
                 width: int = 134, 
                 pool_depth = 3,
                 conv_kernel_size = 3,
                 conv_padding = 1,
                 pool_kernel_size = 2,
                 pool_kernel_stride = 2
                 ):
        super(CNNBlock, self).__init__()

        if pool_depth > len(feature):
            warnings.warn(f"The pool depth has been set to {len(feature)} since it cannot be more", UserWarning)

        self.input_height = height
        self.input_width = width
        self.pool_depth = pool_depth
        self.phases = len(feature) // self.pool_depth
        self.last_phase = len(feature) % self.pool_depth

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_stride)
        self.out_put_size = {}

        iterator = 0
        self.layers = nn.ModuleList()
        while iterator != (len(feature)-1):
            self.layers.append(nn.Conv2d(feature[iterator], feature[iterator+1], kernel_size=conv_kernel_size, padding=conv_padding))
            iterator+=1

        self.batch_norm = nn.BatchNorm2d(feature[len(feature)-1])

        self.out_put_size["features"] = feature[len(feature)-1]

        temp_height = self.input_height
        temp_width = self.input_width
        
        for _ in range(self.phases):
            temp_height = int( ((temp_height - pool_kernel_size)/pool_kernel_stride)+1 )
            temp_width = int( ((temp_width - pool_kernel_size)/pool_kernel_stride)+1 )

        if self.last_phase != 0 :
            temp_height = int( ((temp_height - pool_kernel_size)/pool_kernel_stride)+1 )
            temp_width = int( ((temp_width - pool_kernel_size)/pool_kernel_stride)+1 )


        self.out_put_size["height"] = temp_height
        self.out_put_size["width"] = temp_width
        
        if (self.out_put_size["height"] <= 1) or (self.out_put_size["width"] <= 1):
            warnings.warn(f"The output features are less or equal than 1x1, this makes no sense you fucking moron", UserWarning)
    
    def forward(self, x):
        
        iterator = 0
        for _ in range(self.phases):
            for _ in range(self.pool_depth):
                x =  F.relu(self.layers[iterator](x))
                iterator += 1
            x = self.pool(x)
        
        for _ in range(self.last_phase):
            x =  F.relu(self.layers[iterator](x))
        x = self.pool(x)

        return self.batch_norm(x)


if __name__ == '__main__':
    height=600
    width=600

    model = CNNBlock(height=height, width=width)

    input_tensor = torch.randn(32,1, height, width)  

    output = model(input_tensor)

    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(model.out_put_size)