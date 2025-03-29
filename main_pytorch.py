import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from models.pinn import PINNKS
from utils.plotting import plot_time_series
from utils.training import train_model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for data files and adjust paths if needed
    data_paths = ["../KSSolution.npy", "../KSTime.npy"]
    alt_data_paths = ["KSSolution.npy", "KSTime.npy"]
    
    # Try to find the data files
    if os.path.exists(data_paths[0]):
        solution_path = data_paths[0]
        time_path = data_paths[1]
    elif os.path.exists(alt_data_paths[0]):
        solution_path = alt_data_paths[0]
        time_path = alt_data_paths[1]
    else:
        print("Data files not found. Generating synthetic data for demonstration.")
        # Generate synthetic data
        x_size = 201
        t_size = 2000
        
        # Create synthetic solution (wave-like pattern)
        u_data = torch.zeros((x_size, t_size))
        t_data = torch.linspace(0, 100, t_size)[:, None]
        x = torch.linspace(0, 200, x_size)[:, None]
        
        # Generate a simple wave pattern
        for i in range(t_size):
            u_data[:, i] = 0.5 * torch.sin(0.1 * x.squeeze() + 0.05 * t_data[i]) + \
                          0.3 * torch.sin(0.2 * x.squeeze() - 0.03 * t_data[i])
    else:
        # Load data from files
        print(f"Loading data from {solution_path} and {time_path}")
        u_data = torch.from_numpy(np.load(solution_path).T).float()
        u_data = torch.vstack((u_data, u_data[0:1, :]))
        t_data = torch.from_numpy(np.load(time_path)[:, None]).float()
        x = torch.linspace(0, 200, u_data.shape[0])[:, None]

    # Split data
    train_test_perc = 0.8
    split_idx = int(t_data.size(0) * train_test_perc)
    u_train, u_test = u_data[:, :split_idx], u_data[:, split_idx:]
    t_train, t_test = t_data[:split_idx], t_data[split_idx:]
    
    max_t_train = t_train.max()

    # Generate training points
    num_train_samples = 20000
    generator = torch.Generator().manual_seed(1000)
    
    # Generate collocation points
    tx_eqn = torch.zeros((num_train_samples, 2))
    tx_eqn[:, 0] = (max_t_train - t_data[0]) * torch.rand(num_train_samples, generator=generator) + t_data[0]
    tx_eqn[:, 1] = (x[-1] - x[0]) * torch.rand(num_train_samples, generator=generator)
    
    # Create model and optimizer
    model = PINNKS(hidden_layers=[50]*4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloaders
    train_dataset = TensorDataset(tx_eqn)
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    
    # Train model
    num_epochs = 20
    history = train_model(model, train_loader, optimizer, num_epochs, device)
    
    # Save model
    torch.save(model.state_dict(), 'dense_net_ks.pt')
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        t_flat = t_data.clone()
        x_flat = x.clone()
        T, X = torch.meshgrid(t_flat.squeeze(), x_flat.squeeze(), indexing='ij')
        tx = torch.stack([T.flatten(), X.flatten()], dim=-1).to(device)
        U = model(tx).cpu()
        U = U.reshape(T.shape)

    # Plotting
    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(10, 4))
    plot_time_series(u_data.numpy(), X.numpy(), T.numpy(), fig=fig, ax=ax[0])
    ax[0].set_title("Original")
    ax[0].set_ylabel("Time")
    ax[0].set_xlabel("x")
    ax[0].set_ylim(500, 2000)
    ax[0].set_yticks(np.insert(np.arange(600, 2200, 200), 0, 500))
    
    plot_time_series(U.numpy(), X.numpy(), T.numpy(), fig=fig, ax=ax[1])
    ax[1].set_title("Predicted")
    ax[1].set_xlabel("x")
    
    rel_err = (torch.abs(U - u_data) / torch.abs(u_data + 1.0)).T.numpy()
    rel_err[rel_err > 1.0] = 1.0
    plot_time_series(rel_err, X.numpy(), T.numpy(), fig=fig, ax=ax[2])
    ax[2].set_title("Relative difference")
    ax[2].set_xlabel("x")
    
    plt.show()