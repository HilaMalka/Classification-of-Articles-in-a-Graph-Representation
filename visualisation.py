from data_loader import load_dataset
import matplotlib.pyplot as plt
from dataset import HW3Dataset
import seaborn as sns
import numpy as np
import pandas as pd
import torch

# Module used for visaulisation


# Load Data
dataset = HW3Dataset(root='data/hw3/')
data = load_dataset()


print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print('===========================================================================================================')


# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask_bool.sum()}')
print(f'Training node label rate: {int(data.train_mask_bool.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


#Histogram
def create_class_histogram(data):
    # Count the occurrences of each class label
    class_labels = data.y
    unique_labels, label_counts = np.unique(class_labels, return_counts=True)

    # Create a Seaborn histogram
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_labels, y=label_counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Counts of Class Labels')

    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=90)

    # Display the histogram
    plt.savefig("/home/student/HW3/visualisations/class_count.png")
    plt.show()



def create_neighbor_count_heatmap(data):
    # Extract the class labels and edge index from the data
    class_labels = data.y
    edge_index = data.edge_index

    # Get the unique class labels
    unique_labels = np.unique(class_labels)

    # Create an empty matrix to store neighbor counts
    num_classes = len(unique_labels)
    neighbor_counts = np.zeros((num_classes, num_classes))

    # Count the number of neighbors for each edge
    for i, j in zip(edge_index[0], edge_index[1]):
        label_i = class_labels[i]
        label_j = class_labels[j]
        neighbor_counts[label_i, label_j] += 1

    # Create a heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(neighbor_counts), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=np.arange(num_classes), labels=unique_labels, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=unique_labels)
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.title('Logarithmic Neighbor Counts Heatmap')
    plt.savefig("/home/student/HW3/visualisations/neighbors.png")
    plt.show()


def visualize_self_citation_proportions(data):
    # Extract the class labels and edge index from the data
    class_labels = data.y
    edge_index = data.edge_index

    # Get the unique class labels
    unique_labels = np.unique(class_labels)
    num_classes = len(unique_labels)

    # Create an empty array to store the citation counts
    citation_counts = np.zeros(num_classes)

    # Count the number of citations received by each category
    for i, j in zip(edge_index[1], edge_index[0]):
        label_i = class_labels[i]
        citation_counts[label_i] += 1

    # Create an empty array to store the self-citation proportions
    self_citation_proportions = np.zeros(num_classes)

    # Calculate the proportion of citations from each category's own category
    for i, j in zip(edge_index[1], edge_index[0]):
        label_i = class_labels[i]
        if label_i == class_labels[j]:
            self_citation_proportions[label_i] += 1

    self_citation_proportions /= citation_counts

    # Create a DataFrame to store the self-citation proportions
    df_proportions = pd.DataFrame({'Category': unique_labels, 'Proportion': self_citation_proportions})

    # Sort the DataFrame by the self-citation proportions
    df_proportions = df_proportions.sort_values('Proportion', ascending=False)

    # Create a figure and axis for the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the bar chart
    ax.bar(df_proportions['Category'], df_proportions['Proportion'])

    # Set axis labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Proportion of Self-Citations')
    ax.set_title('Proportion of Self-Citations by Category')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Manually set the x-axis tick positions and labels
    x_ticks = np.arange(num_classes)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    # Show the plot
    plt.savefig("/home/student/HW3/visualisations/self_citations.png")
    plt.show()


# def citatition_by_year(data):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     fields = torch.squeeze(data.y).tolist()
#     years = torch.squeeze(data.node_year).tolist()
#     all_years = sorted(set(years))
#     field_year_dict = {f: {year: 0 for year in range(1970,2019)} for f in fields}

#     for i, j in zip(data.edge_index[0], data.edge_index[1]):
#         # i cited j
#         cited_catagory = fields[j]
#         cited_in_year = years[j]
#         field_year_dict[cited_catagory][cited_in_year] += 1

#     sorted_data = sorted(field_year_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)
#     top_classes = [color_label for color_label, _ in sorted_data[:6]]
#     for catagory, citations in field_year_dict.items():

#         counts = list(citations.values())
#         cited_in = list(citations.keys())
#         ax.plot(cited_in, counts, label=catagory)

#     plt.xlabel('Year')
#     plt.ylabel('Number of citations')
#     plt.title('Number of citation per year')
#     handles, labels = ax.get_legend_handles_labels()
#     selected_handles = [handle for handle, label in zip(handles, labels) if label.split(': ')[1] in top_classes]
#     selected_labels = [label for label in labels if label.split(': ')[1] in top_classes]
#     ax.legend(selected_handles, selected_labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3))
#     plt.tight_layout(rect=[0, 0, 1, 0.9])

#     plt.show()



# Create Plots
def create_vis(data):
    create_class_histogram(data)
    create_neighbor_count_heatmap(data)
    visualize_self_citation_proportions(data)
