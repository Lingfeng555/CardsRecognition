
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.Dense import DenseBlock
from .CardsClassifier import CardClassifier

class SupraCardClassifier(nn.Module):

    def __init__(self, image_size: torch.Size, convolution_structure: list, expert_output_len: int, extractors_output_len: int, expert_depth: int, n_extractor: int, output_len: int, pool_depth: int):
        super(SupraCardClassifier, self).__init__()

        self.extractors = nn.ModuleList([
            CardClassifier(image_size = image_size,
                            convolution_structure = convolution_structure,
                            expert_output_len = expert_output_len,
                            output_len = extractors_output_len,
                            expert_depth = expert_depth,
                            pool_depth=pool_depth
                           ) for _ in range(n_extractor)
        ])

        final_block_structure = self.get_dense_structure(input_size = extractors_output_len * n_extractor, output=output_len)
        self.final_dense_block = DenseBlock(input_size= extractors_output_len * n_extractor, hidden_layers=final_block_structure, output_len=output_len )
        self.batch_norm = nn.BatchNorm1d(extractors_output_len * n_extractor)

    def forward(self, x):
        features = [extractor(x).view(x.size(0), -1) for extractor in self.extractors]
        x = torch.cat(features, dim=1)
        x = self.batch_norm(x)
        x = self.final_dense_block(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def get_dense_structure (self, input_size: int, output: int, stop = 2):
        i = input_size
        ret = []
        while i > output:
            ret.append( i - 10 )
            i = i - 10
            if len(ret) == stop: break
        return ret
    
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    # Crear instancia del modelo
    model = SupraCardClassifier(convolution_structure=[1,8,8,16,16,32,32,64,64,64,64], 
                            image_size=torch.Size((134, 134)), 
                            expert_output_len=3, 
                            extractors_output_len=10,
                            expert_depth=4,
                            output_len=53,
                            n_extractor=2,
                            pool_depth=2
                            )

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(2, 1, 134, 134)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")