
import torch
import torch.nn as nn
import torch.nn.functional as F


from .components.CNNBlock import CNNBlock
from .components.Dense import DenseBlock
from .components.AttentionBlock import AttentionBlock

class CardClassifier(nn.Module):
    
    cnn_block : CNNBlock
    experts : nn.ModuleList
    attention_block : AttentionBlock
    wighted_sum : DenseBlock
    experts_output: torch.tensor
    
    def __init__(self, image_size: torch.Size, convolution_structure: list, expert_output_len: int, output_len: int, expert_depth: int, pool_depth: int):
        super(CardClassifier, self).__init__()
        
        self.cnn_block = CNNBlock(feature=convolution_structure, height=image_size[0], width=image_size[1], pool_depth=pool_depth)
        
        feature_height = self.cnn_block.out_put_size["height"]
        feature_width = self.cnn_block.out_put_size["width"]
        n_features = self.cnn_block.out_put_size["features"]
        
        flatten_feature_size = feature_height * feature_width
        expert_hidden_layers = self.get_dense_structure(input_size=feature_height * feature_width, output=expert_output_len, stop = expert_depth)
        
        self.experts = nn.ModuleList([DenseBlock(output_len=expert_output_len,
                                                 hidden_layers=expert_hidden_layers, 
                                                 input_size=flatten_feature_size
                                                 ) for _ in range(n_features)])
        
        self.attention_block = AttentionBlock(attention_value=1, height=feature_height, width=feature_width, num_features=n_features)
        
        final_weighted_sum_layers = self.get_final_dense_structure(input_size=n_features*expert_output_len, output=output_len, stop = expert_depth)
        
        self.wighted_sum = DenseBlock(input_size=n_features*expert_output_len, hidden_layers=final_weighted_sum_layers, output_len = output_len)

    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        features = self.cnn_block(x)

        attention_values = self.attention_block(features)
        features = features.view(features.shape[0], self.cnn_block.out_put_size["features"], -1)
        x = nn.functional.relu(torch.stack([self.experts[i](features[:, i, :], attention_values[:, i, :]) for i in range(len(self.experts))], dim=1))
        
        self.experts_output = x
        x = x.flatten(start_dim=1)
        
        x = self.wighted_sum(x, att = 1) 
        return x

    def get_dense_structure (self, input_size: int, output: int, stop = 2):
        i = 1
        ret = [input_size]
        while ret[-1]//i > output:
            ret.append( ret[-1] // i )
            i = i * 2
        return ret
    
    def get_final_dense_structure (self, input_size: int, output: int, stop = 2):
        i = input_size
        ret = []
        while i > output:
            ret.append( i - 10 )
            i = i - 10
            if len(ret) == stop: break
        return ret

    def get_expert_output_dict(self)->dict:
        batch, experts, outputs = self.experts_output.size()

        ls = self.experts_output.to("cpu").detach().numpy().tolist()

        ret = {}

        for y in range(experts):
            for z in range(outputs):
                ret[f"expert_{y}_{z}"] = []

        for x in range(batch):
            for y in range(experts):
                for z in range(outputs):
                    ret[f"expert_{y}_{z}"].append(ls[x][y][z])

        return ret
        
if __name__ == "__main__":
    # Crear instancia del modelo
    model = CardClassifier(convolution_structure=[1,8,8,16,16,32,32,64,64,128, 128], image_size=torch.Size((134, 134)), expert_output_len=2, output_len=10)

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(2, 1, 134, 134)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")