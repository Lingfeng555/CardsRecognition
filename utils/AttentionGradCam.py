import torch
import torch.nn.functional as F

class AttentionGradCAM:
    def __init__(self, model, target_layer, attention_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.attention_layer = attention_layer 
        self.gradients = None
        self.activations = None
        self.attention_values = None 

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
        if self.attention_layer is not None:
            self.attention_layer.register_forward_hook(self.save_attention)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def save_attention(self, module, input, output):
        self.attention_values = output.detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * self.activations, dim=1)

        if self.attention_values is not None:
            attention = self.attention_values.unsqueeze(-1)  
            cam = torch.sum(torch.relu(attention * self.activations * weights), dim=1)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        return cam, target_class