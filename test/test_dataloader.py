from dataloaders.complete_dataloader import complete_dataloader
from configs import gcn_roberta_config
from tqdm import tqdm
import os
import networkx as nx
import matplotlib.pyplot as plt

os.chdir("../")

train = complete_dataloader(subset="train", config=gcn_roberta_config, batch_size=32, custom_sampler=True)
train_normal = complete_dataloader(subset="train", config=gcn_roberta_config, batch_size=32)

bar = tqdm(train)

positive = 0
allcnt = 0

for i, sample in enumerate(bar):
    allcnt += len(sample["id"])
    positive += sum(sample["label"])
    if i == 1:
        print(sample["id"])
    """
    nodes = []
    edges = []
    for idx, label in enumerate(sample["label"]):
        nodes.append(idx)
    for i in range(sample["edge_index"].shape[1]):
        edges.append((sample["edge_index"][0, i].item(), sample["edge_index"][1, i].item()))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # Choose a layout
    pos = nx.spring_layout(G)  # Replace with circular_layout, random_layout, shell_layout, or spectral_layout
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    plt.show()
    break
    """

print("Proportion of positive with custom sampler: {}".format(positive / allcnt))

bar = tqdm(train_normal)

positive = 0
allcnt = 0

for i, sample in enumerate(bar):
    allcnt += len(sample["id"])
    positive += sum(sample["label"])

print("Proportion of positive without custom sampler: {}".format(positive / allcnt))
