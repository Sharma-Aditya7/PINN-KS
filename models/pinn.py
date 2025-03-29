import torch
import torch.nn as nn

class PINNKS(nn.Module):
    def __init__(self, hidden_layers, activation=nn.Tanh()):
        super().__init__()
        
        layers = []
        input_dim = 2  # t, x
        
        # Input scaling layers
        self.t_scale = ScaleLayer(1, 10)
        self.x_scale = ScaleLayer(1, 20)
        
        # Hidden layers
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                activation
            ])
            prev_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        t, x = x[:, 0:1], x[:, 1:2]
        t = self.t_scale(t)
        x = self.x_scale(x)
        return self.network(torch.cat([t, x], dim=1))
    
    def pde_loss(self, tx):
        tx.requires_grad_(True)
        u = self(tx)
        
        # Calculate derivatives
        u_t = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0][:, 0:1]
        u_x = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0][:, 1:2]
        u_xx = torch.autograd.grad(u_x, tx, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)[0][:, 1:2]
        u_xxxx = torch.autograd.grad(u_xx, tx, grad_outputs=torch.ones_like(u_xx),
                                   create_graph=True)[0][:, 1:2]
        
        # KS equation: u_t + u*u_x + u_xx + u_xxxx = 0
        f = u_t + u*u_x + u_xx + u_xxxx
        return torch.mean(f**2)

class ScaleLayer(nn.Module):
    def __init__(self, scale_factor, bias_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.bias_factor = bias_factor
        
    def forward(self, x):
        return self.scale_factor * x + self.bias_factor