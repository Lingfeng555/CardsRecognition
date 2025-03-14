import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class DenseBlock(nn.Module):

    input_size: int
    output_len: int
    layers: nn.ModuleList

    def __init__(self, input_size: int, hidden_layers: list, output_len: int):
        super(DenseBlock, self).__init__()

        self.input_size = input_size
        self.output_len = output_len

        hidden_layers.insert(0, input_size)
        hidden_layers.append(output_len)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_layers[i],hidden_layers[i+1]) 
            for i in range(len(hidden_layers)-1)
            ])
    
    def forward(self, x: torch.tensor, att: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x * att
    
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())

        
if __name__ == '__main__':
    model = DenseBlock(input_size=2334, hidden_layers=[543,123,12,4], output_len=1)

    input_tensor = torch.randn(32, 2334)  

    output = model(input_tensor)

    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {model.n_parameters()}")
    print(model.output_len)