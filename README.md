# graph-transformers
Research on Transformers' performance on graph-based tasks


for Graph Representation?
Representation: 
- SHortest path matrix is often directly used as a feature (1), (2), we aim to use it as a target

MODEL:
Input: node tensors, each tensor is the list of edges

Background:
- Transformers
- Review GNN
- review link between gnn and transformers
- review existing literature on graph transformers (Graphormers, etc.)

Benchmarks:
- Open Graph Benchmark, Open Graph Benchmark Large Scale Challenge
- Benchmarking-GNN

(1) https://arxiv.org/pdf/2106.05234.pdf
Do Transformers Really Perform Bad

(2) https://arxiv.org/pdf/1905.12712.pdf
Path-Augmented Graph Transformer Network

Things to test:
- Trying shortest path / coloring number
- Trying MLP, GNN, Transformers
- Attn mask (adjacency matrix)
- Adding T_{in} connections
- Changing number of GNN iterations/Transformer's attention layers (not nb. of heads)
