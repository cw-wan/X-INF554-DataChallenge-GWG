import streamlit as st
import json
import plotly.graph_objects as go
import networkx as nx

MAP_RELATIONS = {
    "Elaboration": "red", "Question-answer_pair": "blue", "Correction": "green", "Contrast": "yellow",
    "Background": "black", "Explanation": "pink",
    "Clarification_question": "orange", "Acknowledgement": "purple", "Conditional": "brown", "Alternation": "gray",
    "Q-Elab": "cyan", "Parallel": "magenta",
    "Continuation": "lime", "Narration": "olive", "Comment": "teal", "Result": "navy"
}

dialog = "ES2002a"
training_labels = json.load(open("data/training_labels.json"))


def get_subgraph(G, index, depth):
    # subgraph should not lose nodes attribute and edges attribute
    subgraph = nx.Graph()
    subgraph.add_node(index)
    subgraph.nodes[index]['label'] = G.nodes[index]['label']
    # use BFS to get the subgraph
    queue = []
    queue.append(index)
    while depth > 0:
        length = len(queue)
        for i in range(length):
            node = queue.pop(0)
            for neighbor in G.neighbors(node):
                if neighbor not in subgraph.nodes:
                    subgraph.add_node(neighbor)
                    subgraph.nodes[neighbor]['label'] = G.nodes[neighbor]['label']
                    subgraph.add_edge(node, neighbor)
                    subgraph.edges[node, neighbor]['label'] = G.edges[node, neighbor]['label']
                    queue.append(neighbor)
        depth -= 1
    return subgraph


def input_dialogue(path):
    G = nx.Graph()
    ## load data
    if (path == ""):
        path = dialog
    dialogue_json = json.load(open("data/training/" + path + ".json"))
    dialogue_txt = open("data/training/" + path + ".txt").readlines()
    # each line in dialogue_txt is an edge with attribute, for example "5 Elaboration 6",so we need to split it,
    # use edge_list to store all the edges
    edge_list = []
    for line in dialogue_txt:
        line = line.split()
        edge_list.append((int(line[0]), int(line[2]), line[1]))

    dialogue_labels = training_labels[path]

    length = len(dialogue_json)
    G.add_nodes_from(range(length))
    # add edges using edge_list [0] and [1]
    G.add_edges_from([(edge[0], edge[1]) for edge in edge_list])
    for edge in edge_list:
        G.edges[edge[0], edge[1]]['label'] = edge[2]
    for i in range(length):
        G.nodes[i]['label'] = dialogue_labels[i]
    if subgraph != "":
        G = get_subgraph(G, subgraph[0], subgraph[1])
    pos = nx.spring_layout(G)
    # draw 16 different colors for 16 different relations in one graph
    edge_traces = []
    for i in range(16):
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if G.edges[edge[0], edge[1]]['label'] == list(MAP_RELATIONS.keys())[i]:
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        edge_trace = go.Scatter(
            legendgroup="Edges",
            legendgrouptitle_text="Relations",
            name=list(MAP_RELATIONS.keys())[i],
            x=edge_x, y=edge_y,
            line=dict(width=2, color=MAP_RELATIONS[list(MAP_RELATIONS.keys())[i]]),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)

    # draw the nodes,nodes have different shapes according to their labels 0 or 1
    node_traces = []
    node_x = []
    node_y = []
    node0_x = []
    node0_y = []
    node1_x = []
    node1_y = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if G.nodes[node]['label'] == 0:
            node0_x.append(x)
            node0_y.append(y)
        else:
            node1_x.append(x)
            node1_y.append(y)

    node_trace0 = go.Scatter(
        legendgroup="Nodes",
        legendgrouptitle_text="Node Importance",
        name="0",
        x=node0_x, y=node0_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='darkslateblue',
            size=6,
            line_width=1,
            symbol='circle'
        ))
    node_trace1 = go.Scatter(
        legendgroup="Nodes",
        name="1",
        x=node1_x, y=node1_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='orangered',
            size=6,
            line_width=1,
            symbol='star'
        ))
    node_trace0.text = [dialogue_json[node]['speaker'] + ": " + dialogue_json[node]['text'] for node in G.nodes() if
                        G.nodes[node]['label'] == 0]
    node_trace1.text = [dialogue_json[node]['speaker'] + ": " + dialogue_json[node]['text'] for node in G.nodes() if
                        G.nodes[node]['label'] == 1]
    node_traces.append(node_trace0)
    node_traces.append(node_trace1)

    # draw the index number of the nodes
    node_index_trace = go.Scatter(
        name="Node Index",
        x=node_x, y=node_y,
        mode='text',
        hoverinfo='text',
        text=[str(node) for node in G.nodes()],
        textposition="top center",
        textfont=dict(
            size=14,
            color="black"
        )
    )

    # draw the graph, legend is the 16 different relations without any nodes
    fig = go.Figure(data=[*edge_traces, *node_traces, node_index_trace],
                    layout=go.Layout(
                        title='<br>Graph of ' + path + str(subgraph),
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    st.plotly_chart(fig)
    # print the text of the dialogue of the subgraph
    st.write("The text of the dialogue:")
    # get the node index in the subgraph, sort it
    nodes = list(G.nodes())
    nodes.sort()
    for node in nodes:
        # index + speaker + text, the order should from 0 to length-1
        st.write(str(node) + " " + dialogue_json[node]['speaker'] + ": " + dialogue_json[node]['text'])


st.title("Graph")
path = st.text_input("Enter the dialogue id, eg: ES2002a")
# input index and depth
subgraph = st.text_input("Enter the index and depth to get the subgraph eg: 10,5")
if subgraph != "":
    subgraph = subgraph.split(",")
    subgraph = [int(i) for i in subgraph]

input_dialogue(path)
