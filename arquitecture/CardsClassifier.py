
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
    device : str
    
    image_height: int
    image_width: int
    convolution_structure: list
    expert_output_len: int
    output_len: int
    
    def __init__(self, image_size: torch.Size, convolution_structure: list, expert_output_len: int, output_len: int, pool_depth: int):
        super(CardClassifier, self).__init__()
        
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.convolution_structure = convolution_structure
        self.expert_output_len = expert_output_len
        self.output_len = output_len
        self.pool_depth = pool_depth
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn_block = CNNBlock(feature=convolution_structure, height=image_size[0], width=image_size[1], pool_depth=pool_depth)
        self.expert_output_len = expert_output_len
        
        feature_height = self.cnn_block.out_put_size["height"]
        feature_width = self.cnn_block.out_put_size["width"]
        n_features = self.cnn_block.out_put_size["features"]
        
        flatten_feature_size = feature_height * feature_width
        expert_hidden_layers = self.get_dense_structure(input_size=feature_height * feature_width, output=expert_output_len)
        
        self.experts = nn.ModuleList([DenseBlock(output_len=expert_output_len,
                                                 hidden_layers=expert_hidden_layers, 
                                                 input_size=flatten_feature_size
                                                 ) for _ in range(n_features)])
        
        self.attention_block = AttentionBlock(attention_value=1, height=feature_height, width=feature_width, num_features=n_features)
        
        final_weighted_sum_layers = self.get_final_dense_structure(input_size=n_features*expert_output_len, output=output_len)
        
        self.wighted_sum = DenseBlock(input_size=n_features*expert_output_len, hidden_layers=final_weighted_sum_layers, output_len = output_len)

    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        batch, _, _, _ = x.size()
        features = self.cnn_block(x)

        attention_values = self.attention_block(features)
        features = features.view(features.shape[0], self.cnn_block.out_put_size["features"], -1)
        
        x = nn.functional.relu(
            torch.stack(
                [
                    self.experts[i](features[:, i, :], attention_values[:, i, :])
                    if not isinstance(self.experts[i], nn.Identity)
                    else torch.zeros(batch, self.expert_output_len, device=self.device)
                    for i in range(len(self.experts))
                ],
                dim=1
            )
        )
        self.experts_output = x
        x = x.flatten(start_dim=1)
        
        x = self.wighted_sum(x, att = 1) 
        return x

    def get_dense_structure (self, input_size: int, output: int):
        i = 1
        ret = [input_size]
        while ret[-1]//i > output:
            ret.append( ret[-1] // i )
            i = i * 2
        return ret
    
    def get_final_dense_structure (self, input_size: int, output: int):
        i = input_size
        ret = [input_size]
        while ret[-1]//i > output:
            ret.append( ret[-1] // i )
            i = i + 2
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
    
    def prune_experts(self, list_of_experts: list) -> None:
        '''Receive a list of indexes of experts'''
        for expert_index in list_of_experts:
            self.experts[expert_index] = nn.Identity().to(self.device)
            
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