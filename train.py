import torch
from graphVAE import GraphVAE
from utils import loss_function
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def train(model, optimizer, loader,device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data.x, data.edge_index)
        loss = loss_function(recon_x, data.x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader,device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon_x, mu, logvar = model(data.x, data.edge_index)
            loss = loss_function(recon_x, data.x, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(loader)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path=os.getcwd()
    dataset = TUDataset(root=os.path.join(path,'ENZYMES'), name='ENZYMES')
    indices = list(range(len(dataset)))  
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)  

    train_dataset = torch.utils.data.Subset(dataset, train_idx)  
    test_dataset = torch.utils.data.Subset(dataset, test_idx)  

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = GraphVAE(dataset.num_features, 64).to(device)  

    # Load model if available
    model_files = sorted([file for file in os.listdir() if file.endswith('.pth')])
    if model_files:
        model.load_state_dict(torch.load(model_files[-1], map_location=device))
        print(f'Loaded model from {model_files[-1]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(2000):
        train_loss = train(model, optimizer, train_loader, device)
        if (epoch+1) % 20 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')

        if (epoch+1) % 1000 == 0:
            test_loss = test(model, test_loader, device)
            print(f'Epoch: {epoch+1}, Test Loss: {test_loss}')

        if (epoch+1) % 2000 == 0:
            torch.save(model.state_dict(), f'model_{epoch+1}.pth')


if __name__ == "__main__":
    main()