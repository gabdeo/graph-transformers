import torch
import sys
import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_transformers import Transformer
from graph_transformers.models.gnn import GNN, create_synthetic_data, matrix_to_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_perform_transformer_test_cases():
    num_tokens = 100
    batch_size = 10
    dim = 64
    num_layers = 4
    num_heads = 2
    dummy_model = Transformer(
        dim=dim, attn_dim=32, mlp_dim=dim, num_heads=num_heads, num_layers=num_layers
    ).to(device)

    inp = torch.randn(batch_size, num_tokens, dim, device=device)

    # test case 1 regular forward pass
    print("Test Case 1")
    with torch.no_grad():
        output, alpha = dummy_model(inp, attn_mask=None)
        assert alpha is None
        assert output.shape == (
            batch_size,
            num_tokens,
            dim,
        ), f"wrong output shape {output.shape}"

    # test case 2 collect attentions
    print("Test Case 2")
    with torch.no_grad():
        output, alpha = dummy_model(inp, attn_mask=None, return_attn=True)
        assert output.shape == (
            batch_size,
            num_tokens,
            dim,
        ), f"wrong output shape {output.shape}"
        assert alpha.shape == (
            batch_size,
            num_layers,
            num_heads,
            num_tokens,
            num_tokens,
        ), f"wrong alpha shape {alpha.shape}"

    # test case 3 with attention mask
    attn_mask = torch.zeros(batch_size, num_tokens, num_tokens, device=device)
    attn_mask[:, torch.arange(num_tokens), torch.arange(num_tokens)] = 1
    attn_mask[:, torch.arange(num_tokens)[1:], torch.arange(num_tokens)[:-1]] = 1

    with torch.no_grad():
        output, alpha = dummy_model(inp, attn_mask=attn_mask, return_attn=True)
        print("Attention mask pattern", attn_mask[0])
        print("Alpha pattern", alpha[0, 0, 0])
        assert torch.all(alpha.permute(1, 2, 0, 3, 4)[:, :, attn_mask == 0] == 0).item()


def test_perform_gnn_test_cases():
    SETTINGS = {
        "num_iter": 4,  # For a graph with 4 nodes
        "message_depth": 3,
        "message_breadth": 16,
        "update_depth": 2,
        "update_breadth": 8,
        "use_x_i": True,
        "aggr": "min",
    }
    dummy_model = GNN(**SETTINGS).to(device)
    num_nodes = 4
    batch_size = 10
    inp = matrix_to_graph(create_synthetic_data(batch_size, num_nodes), batch_size)

    # test case 1 regular forward pass
    print("Test Case 1")
    with torch.no_grad():
        output = dummy_model(inp)
        assert output.shape == (
            batch_size,
            num_nodes,
        ), f"wrong output shape {output.shape}"
