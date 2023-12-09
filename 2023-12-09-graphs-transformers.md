---
layout: distill
title: Graph Transformers
description: This is a project on Transformers architecture. We study how transformers learn fundamental graph properties.
date: 2023-12-09
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2023-12-09-graphs-transformers.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Overview
  - name: A measure of learnability
    subsections:
    - name: Algorithmic Alignment
    - name: Application to Transformers
  - name: Graph Transformer Model Design
    subsections:
    - name: Input Tokenization Strategy
    - name: Attention Mechanism: Query-Key-Value
    - name: Model Architecture Overview
    - name: Baseline Model: Multilayer Perceptron (MLP)
    - name: Advanced Benchmark: Graph Neural Network (GNN)
  - name: Methodology for Training and Evaluation
    subsections:
    - name: Constructing the Dataset
    - name: Training Protocols
    - name: Metrics and Evaluation Criteria
  - name: Results and Comparative Analysis


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Project Overview

Transformers perform well in domains that benefit from their ability to understand long-range dependencies and contextual information. 
While their main applications are targeting natural language processing<d-cite key="DBLP:conf/naacl/DevlinCLT19"></d-cite>
, computer vision<d-cite key="DBLP:conf/iclr/DosovitskiyB0WZ21"></d-cite>
, or speech recognition<d-cite key="DBLP:conf/icassp/WangWFY20"></d-cite>
, there have been a few recent works taking a closer look at the ability of transformers to learn tasks such as arithmetic, GCD computations, and matrix operations <d-cite key="DBLP:journals/corr/abs-2112-01898"></d-cite><d-cite key="charton2023transformers"></d-cite><d-cite key="lample2019deep"></d-cite>, which has provided a bit of insight into what these transformers have been learning. 

In this project, we are dedicated to exploring the capability of transformers in understanding fundamental graph properties. Our primary focus centers on the Shortest Path Problem, recognized as a classic challenge in both graph theory and Dynamic Programming. To this end, we have developed a specialized Graph Transformer architecture, specifically tailored for this task. Our objective is to assess its effectiveness in terms of accuracy, setting our findings in context by comparing the performance of our Graph Transformer with basic models such as Multilayer Perceptrons (MLPs) and more advanced, state-of-the-art Deep Learning benchmarks like Graph Neural Networks (GNNs).


<div class="row align-items-center mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-graphs-transformers/Critical_1000-vertex_Erdős–Rényi–Gilbert_graph.svg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-graphs-transformers/transformer-architecture-diagram.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



## Dataset

For training and evaluating our different models, we generate a comprehensive dataset comprising 50,000 samples, each representing a graph. These graphs were randomly created following the Erdős–Rényi model, specifically the \( \mathcal{G}(n, p) \) variant, where `n` represents the number of nodes and `p` is the probability of edge formation between any two nodes. In our dataset, each graph consists of 10 nodes (`n = 10`), and the edge probability (`p`) is set at 0.5. This setting ensures a balanced mix of sparsely and densely connected graphs, providing a robust testing ground for the Graph Transformer's ability to discern and compute shortest paths under varied connectivity scenarios.

Furthermore, we assign to the edges in these graphs some weights that are integral values ranging from 1 to 10. This range of weights introduces a second layer of complexity to the shortest path calculations, as the Graph Transformer must now navigate not only the structure of the graph but also weigh the cost-benefit of traversing various paths based on these weights. The inclusion of weighted edges makes the dataset more representative of real-world graph problems, where edges often have varying degrees of traversal difficulty or cost associated with them.

This dataset is designed to challenge and evaluate the Graph Transformer's capability in accurately determining the shortest path in diverse graph structures under different weight conditions. The small number of nodes ensures a wide variability in the degree of connectivity in a sample graph. It also allows for an initial performance evaluation on smaller-scale problems, with the potential to extend these studies to larger-scale graphs in the future. Hence, the dataset's structure supports a comprehensive assessment of the model's performance and its adaptability to a wide range of graph-related scenarios.


## Model design

### Tokenization

The power of transformers architectures compared to GNNs is that it can aggregate global information via the self-attention mechanism, by allowing each node to attend for potentially every other element, and hence communicate with “distant“ nodes – i.e. not in its neighborhood. 
On the contrary, GNNs aggregate information on a localized manner, passing information successively from a node to its neighbors.
Here we propose a method for encoding graphs as a set tokens that would be adapted to transformer architectures. 
For graphs without weights (i.e. our Erdős–Rényi graph), with $n$ nodes:

-	Transform each node in a token 

-	Each token $i$ has $N$ binary components indicating whether the node $i$ is connected to node $j$ – put 1 on component $i$.

For graphs with weights on the edges, a new difficulty is to encode the values of the weights
We can use a similar approach to the encoding of matrices coefficients, for example using base 10 positional encoding (P10) <d-cite key="DBLP:journals/corr/abs-2112-01898"></d-cite>, if we assume that a weight of 0 is equivalent to no connection between the nodes. Otherwise we could still add an additional component to the  token first specifying if an edge exists between nodes $i$ and $j$.

For example an edge of value 7 could be represented as `[1, +, 7, 0, 0, E-2]` where the first component 1 indicates the presence of and edge. Similarly, an absence of edge would be encoded `[0, +, 0, 0, 0, E-2]`.

Note that the tokens here would be representing the edges rather than directly the nodes and their connections to the others.

<blockquote>

Remark: For the problems we are proposing to consider, it seems that a specific ordering of the nodes isn't required. However, for problems that require considering sequence data, we could introduce a node ordering and encode positional information of the nodes. This can be achieved, for example, using graph kernels <d-cite key="DBLP:journals/corr/abs-2106-05667"></d-cite>

</blockquote>


### Architecture

We are not sure what model size to aim for. In <d-cite key="lample2019deep"></d-cite>, the transformer is trained with 8 attention heads, 6 layers, and a dimensionality of 512. This may be a bit ambitious given our resources, so we are exploring different options for model size. It's essential to vary these hyperparameters to gain insights into their impact. The size of our model will also determine the maximum size of our input, which will always be $$n^2$$ (where $n$ is the number of nodes in the graph).

## Evaluation and Comparaison

To gauge the effectiveness of our approach, we plan to conduct model evaluations, comparing the performance of our transformer-based method to established benchmarks, and particularly GNNs. This analysis will allow us to understand the comparative strengths of transformers and how they approach these graph problems in comparison to GNNs (and potentially other models - please feel free to suggest any).