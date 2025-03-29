import torch
from tqdm import tqdm

def train_model(model, train_loader, optimizer, num_epochs, device):
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                tx = batch[0].to(device)
                optimizer.zero_grad()
                
                loss = model.pde_loss(tx)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
    return history