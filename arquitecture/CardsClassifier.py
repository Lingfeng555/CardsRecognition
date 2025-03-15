
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
    
    def __init__(self, image_size: torch.Size, convolution_structure: list, expert_output_len: int, output_len: int, expert_depth: int):
        super(CardClassifier, self).__init__()
        
        self.cnn_block = CNNBlock(feature=convolution_structure, height=image_size[0], width=image_size[1], pool_depth=2)
        
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
        
        final_weighted_sum_layers = self.get_dense_structure(input_size=n_features*expert_output_len, output=output_len, stop = expert_depth)
        
        self.wighted_sum = DenseBlock(input_size=n_features*expert_output_len, hidden_layers=final_weighted_sum_layers, output_len = output_len)

    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        features = self.cnn_block(x)

        attention_values = self.attention_block(features)
        #print(features.shape)
        features = features.view(features.shape[0], self.cnn_block.out_put_size["features"], -1)
        #print("Features.shape (batch_size, n_features, dataflatten) :",features.shape)
        #print("Attention.shape (batch_size, n_features, ,attention value) :", attention_values.shape)
        x = nn.functional.relu(torch.stack([self.experts[i](features[:, i, :], attention_values[:, i, :]) for i in range(len(self.experts))], dim=1))

        x = x.flatten(start_dim=1)
        
        #self.experts_output = x #Store the current output to store it

        #print(f"MoE output {x.shape}")
        x = self.wighted_sum(x, att = 1) # here we dont have any attention
        # x = torch.clamp(x, min=1.0,max=100) #activarlp solo para el modelo entrenado
        return x

    def get_dense_structure (self, input_size: int, output: int, stop = 2):
        i = 1
        ret = [input_size]
        while (input_size // i) > output:
            ret.append( input_size // i )
            i *= 2
            if len(ret) == stop: break
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