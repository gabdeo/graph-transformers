import torch
import tqdm
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from graph_transformers import Transformer, GraphDataset, AverageMeter
from matplotlib import pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def load_model(model_path, **transformer_params):
    # Initialize the model (make sure it's the same architecture as the one trained)
    model = Transformer(**transformer_params)  # Fill in the appropriate parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def train_base_transformer():
    
    # Dataset params
    n = 10  # Number of nodes in the graph
    channels = 2 # Number of edge features

    # Model params
    dimension = n * channels  # Model dimension
    attn_dim = 64  # Model dimension
    mlp_dim = 64 # Hidden layer dimension
    n_heads = 8  # Number of transformer heads
    num_layers = 6  # Number of transformer layers
    train_test_split = 0.6  # Train-test split ratio
    seed = 42  # Seed for reproducibility
    batch_size = 128  # Number of samples per batch
    epochs = 64

    torch.manual_seed(seed)
    model = Transformer(dim=dimension, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=n_heads, num_layers=num_layers, seq_len = n, out_dim = 1).to(device)

    loss_fn = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Generate some random data
    dataset = GraphDataset(num_samples=50000, num_nodes=n, edge_prob=0.3, target_type="shortest_path", max_weight=10, graph_neg_weights=False, seed=seed)
    # plt.hist(dataset.targets, bins = 50)

    # Train-test split
    train_size = int(train_test_split * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    attn_mask = torch.ones((n, n))

    train_loss_hist = []
    val_loss_hist = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # loss_meter = AverageMeter()
        # acc_meter = AverageMeter()
        loss = train_loop(train_dataloader, model, attn_mask, loss_fn, optimizer, device, val_set = val_dataloader)
        if val_dataloader is not None:
            val_loss, val_acc = evaluate(val_dataloader, model, attn_mask, loss_fn, device)

        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss)

        print(f"Train loss: {loss:>7f}, Val loss: {val_loss:>7f}, Val acc: {val_acc:>7f}")

    # Test the model
    print("\n----------\nTest Results:")
    evaluate(test_dataloader, model, attn_mask, loss_fn, device)

    plt.plot(train_loss_hist, label = "Train Loss")
    plt.plot(val_loss_hist, label = "Val Loss")

    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model, dataset




def train_loop(dataloader, model, attn_mask, loss_fn, optimizer, device, val_set = None):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        batch_size = X.shape[0]
        attn_mask_batched = attn_mask.repeat(batch_size, 1, 1)

        X = X.permute((0, 2, 1, 3)).flatten(2, 3).type(torch.float32)
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred, _ = model(X, attn_mask_batched)
        loss = loss_fn(pred, y.type(torch.float32))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss

def evaluate(dataloader, model, attn_mask, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            batch_size = X.shape[0]
            attn_mask_batched = attn_mask.repeat(batch_size, 1, 1)
            
            X = X.permute((0, 2, 1, 3)).flatten(2, 3).type(torch.float32)
            X, y = X.to(device), y.to(device)
            pred, _ = model(X, attn_mask_batched)
            test_loss += loss_fn(pred, y.type(torch.float32)).item()
            correct += (pred.round() == y).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    print(f"Accuracy: {correct:>8f} \n")

    return test_loss, correct

if __name__ == "__main__":
    model, dataset = train_base_transformer()
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    exit(1)