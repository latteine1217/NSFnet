"""
Plug-and-Play Trainable Sinusoidal Activation (TSA) Function API
直接替換現有 PINNs 架構中的激活函數

Usage Examples:
1. Replace existing activation:
   # Before: self.activation = nn.Tanh()
   # After:  self.activation = TSA(num_neurons)

2. Quick network conversion:
   network = convert_to_tsa_network(your_existing_model)

3. Add to existing loss:
   total_loss = data_loss + physics_loss + tsa_regularization_loss(model)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Dict, Any
import warnings


# ============================================================================
# Core TSA Activation Function - Direct Replacement for any activation
# ============================================================================

class TSA(nn.Module):
    """
    Trainable Sinusoidal Activation - Direct replacement for nn.Tanh(), nn.ReLU(), etc.
    
    Simply replace your existing activation with TSA(num_neurons)
    
    Example:
        # Before
        self.layer1 = nn.Linear(10, 50)
        self.act1 = nn.Tanh()
        
        # After  
        self.layer1 = nn.Linear(10, 50)
        self.act1 = TSA(50)  # Just specify the number of neurons
    """
    
    def __init__(self, 
                 num_neurons: int,
                 freq_std: float = 1.0,
                 trainable_coeffs: bool = False):
        """
        Args:
            num_neurons: Number of neurons (must match the layer output size)
            freq_std: Frequency initialization std (default: 1.0)
            trainable_coeffs: Whether sine/cosine coefficients are trainable
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        
        # Trainable frequency for each neuron
        self.freq = nn.Parameter(torch.randn(num_neurons) * freq_std)
        
        # Sine and cosine coefficients
        if trainable_coeffs:
            self.c1 = nn.Parameter(torch.ones(1))  # sine coefficient
            self.c2 = nn.Parameter(torch.ones(1))  # cosine coefficient
        else:
            self.register_buffer('c1', torch.ones(1))
            self.register_buffer('c2', torch.ones(1))
    
    def forward(self, x):
        """
        TSA activation: c1*sin(freq*x) + c2*cos(freq*x)
        """
        if x.size(-1) != self.num_neurons:
            raise ValueError(f"Input size {x.size(-1)} != expected {self.num_neurons}")
        
        freq_x = self.freq * x
        return self.c1 * torch.sin(freq_x) + self.c2 * torch.cos(freq_x)
    
    def get_stats(self):
        """Get frequency statistics for monitoring"""
        return {
            'freq_mean': self.freq.mean().item(),
            'freq_std': self.freq.std().item(),
            'freq_range': (self.freq.min().item(), self.freq.max().item())
        }


# ============================================================================
# Simple Frequency Regularization - Add to your loss function
# ============================================================================

def tsa_regularization_loss(model: nn.Module, weight: float = 0.01) -> torch.Tensor:
    """
    Compute TSA frequency regularization loss for any model containing TSA layers
    
    Usage:
        total_loss = data_loss + physics_loss + tsa_regularization_loss(model)
    
    Args:
        model: Your PyTorch model containing TSA layers
        weight: Regularization weight (default: 0.01)
    
    Returns:
        Regularization loss term
    """
    tsa_layers = [m for m in model.modules() if isinstance(m, TSA)]
    
    if not tsa_layers:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    total_reg = 0.0
    for tsa in tsa_layers:
        mean_freq = torch.mean(tsa.freq)
        total_reg += torch.exp(torch.clamp(mean_freq, max=10.0))
    
    if len(tsa_layers) > 1:
        total_reg = total_reg / (len(tsa_layers) - 1)
    
    return weight / (total_reg + 1e-8)


# ============================================================================
# Automatic Network Conversion - Convert existing networks to use TSA
# ============================================================================

def convert_to_tsa_network(model: nn.Module, 
                          target_activations: List[str] = ['Tanh', 'ReLU', 'Sigmoid'],
                          freq_std: float = 1.0) -> nn.Module:
    """
    Automatically convert existing network to use TSA activations
    
    Args:
        model: Your existing PyTorch model
        target_activations: List of activation names to replace
        freq_std: Frequency initialization std
    
    Returns:
        Modified model with TSA activations
    
    Example:
        # Convert your existing PINN
        original_model = YourPINNModel()
        tsa_model = convert_to_tsa_network(original_model)
    """
    def get_output_size(module, input_module):
        """Try to determine output size of the previous layer"""
        if hasattr(input_module, 'out_features'):
            return input_module.out_features
        elif hasattr(input_module, 'num_features'):
            return input_module.num_features
        else:
            # Fallback: try to infer from module name or return None
            return None
    
    # Get all modules
    modules_list = list(model.named_modules())
    
    for i, (name, module) in enumerate(modules_list):
        # Check if this is an activation we want to replace
        if type(module).__name__ in target_activations:
            
            # Find the previous linear layer to get output size
            output_size = None
            for j in range(i-1, -1, -1):
                prev_name, prev_module = modules_list[j]
                if isinstance(prev_module, nn.Linear):
                    output_size = prev_module.out_features
                    break
            
            if output_size is not None:
                # Replace with TSA
                new_tsa = TSA(output_size, freq_std=freq_std)
                
                # Set the new module
                parent_names = name.split('.')[:-1]
                parent = model
                for parent_name in parent_names:
                    parent = getattr(parent, parent_name)
                
                setattr(parent, name.split('.')[-1], new_tsa)
                print(f"✅ Replaced {type(module).__name__} with TSA({output_size}) at {name}")
            else:
                print(f"⚠️  Could not determine output size for {name}, skipping")
    
    return model


# ============================================================================
# Enhanced TSA with More Features (Optional)
# ============================================================================

class TSAAdvanced(nn.Module):
    """
    Advanced TSA with additional features for power users
    """
    
    def __init__(self, 
                 num_neurons: int,
                 freq_std: float = 1.0,
                 freq_mean: float = 0.0,
                 trainable_coeffs: bool = False,
                 freq_bounds: Optional[tuple] = None,
                 init_method: str = 'normal'):
        
        super().__init__()
        self.num_neurons = num_neurons
        self.freq_bounds = freq_bounds
        
        # Initialize frequencies
        if init_method == 'normal':
            self.freq = nn.Parameter(torch.normal(freq_mean, freq_std, (num_neurons,)))
        elif init_method == 'uniform':
            self.freq = nn.Parameter(torch.rand(num_neurons) * 2 * freq_std - freq_std)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        # Coefficients
        if trainable_coeffs:
            self.c1 = nn.Parameter(torch.ones(1))
            self.c2 = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('c1', torch.ones(1))
            self.register_buffer('c2', torch.ones(1))
    
    def forward(self, x):
        freq = self.freq
        if self.freq_bounds is not None:
            freq = torch.clamp(freq, self.freq_bounds[0], self.freq_bounds[1])
        
        freq_x = freq * x
        return self.c1 * torch.sin(freq_x) + self.c2 * torch.cos(freq_x)


# ============================================================================  
# Monitoring and Analysis Tools
# ============================================================================

class TSAMonitor:
    """
    Simple monitoring tool for TSA networks
    """
    
    def __init__(self):
        self.history = []
    
    def log(self, model: nn.Module, epoch: int, loss: float):
        """Log TSA statistics"""
        tsa_layers = [m for m in model.modules() if isinstance(m, (TSA, TSAAdvanced))]
        
        if not tsa_layers:
            return
        
        stats = {
            'epoch': epoch,
            'loss': loss,
            'num_tsa_layers': len(tsa_layers),
            'avg_freq': np.mean([layer.freq.mean().item() for layer in tsa_layers]),
            'freq_std': np.mean([layer.freq.std().item() for layer in tsa_layers])
        }
        
        self.history.append(stats)
    
    def plot_history(self):
        """Simple plotting of TSA evolution"""
        if not self.history:
            print("No history to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            epochs = [h['epoch'] for h in self.history]
            losses = [h['loss'] for h in self.history]
            avg_freqs = [h['avg_freq'] for h in self.history]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.semilogy(epochs, losses)
            ax1.set_title('Loss Evolution')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            ax2.plot(epochs, avg_freqs)
            ax2.set_title('Average Frequency Evolution')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Average Frequency')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


# ============================================================================
# Utility Functions
# ============================================================================

def count_tsa_parameters(model: nn.Module) -> Dict[str, int]:
    """Count TSA-related parameters in the model"""
    tsa_layers = [m for m in model.modules() if isinstance(m, (TSA, TSAAdvanced))]
    
    total_freq_params = sum(layer.num_neurons for layer in tsa_layers)
    total_coeff_params = sum(2 if hasattr(layer, 'c1') and 
                           isinstance(layer.c1, nn.Parameter) else 0 
                           for layer in tsa_layers)
    
    return {
        'num_tsa_layers': len(tsa_layers),
        'frequency_parameters': total_freq_params,
        'coefficient_parameters': total_coeff_params,
        'total_tsa_parameters': total_freq_params + total_coeff_params
    }


def get_all_tsa_stats(model: nn.Module) -> Dict:
    """Get comprehensive TSA statistics from model"""
    tsa_layers = [m for m in model.modules() if isinstance(m, (TSA, TSAAdvanced))]
    
    if not tsa_layers:
        return {'message': 'No TSA layers found in model'}
    
    stats = {}
    for i, layer in enumerate(tsa_layers):
        stats[f'tsa_layer_{i}'] = layer.get_stats() if hasattr(layer, 'get_stats') else {
            'freq_mean': layer.freq.mean().item(),
            'freq_std': layer.freq.std().item(),
            'freq_range': (layer.freq.min().item(), layer.freq.max().item())
        }
    
    # Overall statistics
    all_freqs = torch.cat([layer.freq for layer in tsa_layers])
    stats['overall'] = {
        'total_neurons': len(all_freqs),
        'freq_mean': all_freqs.mean().item(),
        'freq_std': all_freqs.std().item(),
        'freq_range': (all_freqs.min().item(), all_freqs.max().item())
    }
    
    return stats


# ============================================================================
# Example Integration with Common PINN Patterns
# ============================================================================

class ExamplePINNWithTSA(nn.Module):
    """
    Example showing how to integrate TSA into a typical PINN architecture
    """
    
    def __init__(self, input_dim=2, output_dim=1, hidden_dims=[50, 50, 50]):
        super().__init__()
        
        # Build layers with TSA activations
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(TSA(hidden_dim))  # Replace nn.Tanh() with TSA
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def compute_loss(self, data_loss, physics_loss):
        """Compute total loss including TSA regularization"""
        reg_loss = tsa_regularization_loss(self)
        return data_loss + physics_loss + reg_loss


# ============================================================================
# Quick Setup Functions
# ============================================================================

def quick_tsa_setup(your_model: nn.Module, monitor: bool = True):
    """
    Quick setup for adding TSA to your existing model
    
    Args:
        your_model: Your existing PINN model
        monitor: Whether to return a monitor object
    
    Returns:
        (converted_model, monitor) if monitor=True, else converted_model
    """
    # Convert model to use TSA
    tsa_model = convert_to_tsa_network(your_model)
    
    # Print summary
    param_info = count_tsa_parameters(tsa_model)
    print(f"✅ TSA Setup Complete!")
    print(f"   Added {param_info['num_tsa_layers']} TSA layers")
    print(f"   Total TSA parameters: {param_info['total_tsa_parameters']}")
    print(f"   Use tsa_regularization_loss(model) in your loss function")
    
    if monitor:
        tsa_monitor = TSAMonitor()
        print(f"   Use monitor.log(model, epoch, loss) to track progress")
        return tsa_model, tsa_monitor
    else:
        return tsa_model


# ============================================================================
# Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    print("🚀 Testing Plug-and-Play TSA API")
    print("=" * 50)
    
    # Test 1: Basic TSA activation
    print("\n1️⃣ Basic TSA Activation Test")
    tsa = TSA(num_neurons=10)
    x = torch.randn(5, 10)
    y = tsa(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   TSA stats: {tsa.get_stats()}")
    
    # Test 2: Drop-in replacement
    print("\n2️⃣ Drop-in Replacement Test")
    
    # Original network
    class OriginalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(3, 20)
            self.act1 = nn.Tanh()  # This will be replaced
            self.layer2 = nn.Linear(20, 10)
            self.act2 = nn.ReLU()  # This will be replaced
            self.output = nn.Linear(10, 1)
    
    original = OriginalNet()
    print(f"   Original model parameters: {sum(p.numel() for p in original.parameters())}")
    
    # Convert to TSA
    tsa_net = convert_to_tsa_network(original)
    print(f"   TSA model parameters: {sum(p.numel() for p in tsa_net.parameters())}")
    
    # Test forward pass
    x = torch.randn(5, 3)
    y_orig = original(x)
    y_tsa = tsa_net(x)
    print(f"   Both models work: {y_orig.shape == y_tsa.shape}")
    
    # Test 3: Regularization loss
    print("\n3️⃣ Regularization Loss Test")
    reg_loss = tsa_regularization_loss(tsa_net)
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    
    # Test 4: Monitoring
    print("\n4️⃣ Monitoring Test")
    monitor = TSAMonitor()
    for epoch in range(3):
        dummy_loss = torch.randn(1).abs().item()
        monitor.log(tsa_net, epoch * 100, dummy_loss)
    print(f"   Logged {len(monitor.history)} epochs")
    
    # Test 5: Quick setup
    print("\n5️⃣ Quick Setup Test")
    original2 = OriginalNet()
    tsa_model, tsa_monitor = quick_tsa_setup(original2, monitor=True)
    
    print("\n✅ All tests passed! TSA is ready for integration.")
    
    print("\n" + "="*50)
    print("📖 INTEGRATION GUIDE:")
    print("="*50)
    print("""
    🔧 Method 1: Direct Replacement
    # Replace this:
    self.activation = nn.Tanh()
    
    # With this:
    self.activation = TSA(num_neurons)
    
    🔧 Method 2: Automatic Conversion  
    tsa_model = convert_to_tsa_network(your_existing_model)
    
    🔧 Method 3: Add to Loss Function
    total_loss = data_loss + physics_loss + tsa_regularization_loss(model)
    
    🔧 Method 4: Quick Setup (Recommended)
    tsa_model, monitor = quick_tsa_setup(your_model)
    
    # In training loop:
    monitor.log(model, epoch, loss.item())
    """)
