import pandas as pd
import numpy as np
import torch
import networkx as nx
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
from multiprocessing import Pool, Manager
import concurrent.futures
import time
import os
import random


# Initialize and train the GraphSAGE model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, num_features):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, 16, normalize=False)
        self.conv2 = SAGEConv(16, num_features, normalize=False)

    def forward(self, x, edge_index):
        edge_index = edge_index.squeeze(0)  # Squeeze the batch dimension
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
###############################################################################
def is_in_distribution(model, edge_index, input_row,mean, std,num_services, num_features, threshold=0.1):
    # model.eval()
    with torch.no_grad():
        input_row = torch.tensor(input_row, dtype=torch.float32).view(1, num_services, num_features)
        input_row = (input_row - mean) / std  # Z-score normalization

        # Reconstruct the input row using the trained model
        reconstruction = model(input_row, edge_index)

        # Calculate mean squared error (MSE)
        mse = F.mse_loss(reconstruction, input_row).item()

        # Check if MSE is below the threshold
        in_distribution = mse < threshold

    return in_distribution
##############################################################

# Function to replace anomaly columns with mean values
def replace_anomalies(args):
    # Record the start time
    start_time = time.time()

    model, edge_index, row_id,input_row, mean, std, num_services, num_features, threshold, rootcause = args

    anomaly_detection=False
    true_rootcauses = False
    local_number_of_rootcauses=0
    List_of_candidates =[]

    # anomalyDetection
    output=is_in_distribution(model, edge_index, input_row, mean, std, num_services, num_features, threshold)

    if not output:
        anomaly_detection=True
       #rootcause:
        for i in range(len(input_row)):
            # Create new_row for each column
            new_row = input_row.copy()
            new_row[i] = mean[i]  # Replace the column value with mean

            # # Calculate parameters for the gamma distribution
            # k = (mean[i] / std[i]) ** 2  # shape parameter
            # theta = std[i] ** 2 / mean[i]  # scale parameter
            # # Generate a random number from a gamma distribution
            # new_row[i] = np.random.gamma(k, theta)

            # # Convert mean and std deviation to the scale of the log-normal distribution
            # mu = np.log(mean[i] ** 2 / np.sqrt((std[i] ** 2) + mean[i] ** 2))
            # sigma = np.sqrt(np.log(1 + (std[i] ** 2) / mean[i] ** 2))
            # new_row[i] = lognorm.rvs(s=sigma, scale=np.exp(mu), size=1)

            # Call is_in_distribution
            is_in_dist = is_in_distribution(model, edge_index, new_row, mean, std, num_services, num_features, threshold)

            if  is_in_dist:
                local_number_of_rootcauses+=1
                List_of_candidates.append(i)
                if i == rootcause:  # known root cause
                    true_rootcauses=True

    print(f"Done processing {row_id}th row")
    # Acquire the lock before updating shared variables
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    return anomaly_detection,true_rootcauses,local_number_of_rootcauses,List_of_candidates,elapsed_time

####################################################################


if __name__ == '__main__':

    # Load the data
    data = pd.read_csv('no-interference.csv')

    num_train_data = len(data)
    data = data.sample(n=num_train_data, random_state=np.random.randint(0, num_train_data))

    # Number of services (nodes) and size of the feature vector for each service
    num_services = len(data.columns) - 1
    num_features = 1  # number of features for each service

    # Extract features and labels
    features = data.iloc[:, 1:].values  # Features (latency values for each service)
    features = np.array(features)
    features = features.reshape((data.shape[0], num_services,
                                 num_features))  # Reshape to (number_of_samples, number_of_nodes, number_of_features)

    # Construct the directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_services))

    # update for reach benchmark
    relationships = [
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 7),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 8),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 9),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 10),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 11),
        (0, 1), (1, 2), (2, 5),
        (0, 1), (1, 3)
    ]

    # Add edges to the graph based on relationships
    graph.add_edges_from(relationships)

    # Get edge indices as a PyTorch tensor
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_index = edge_index.view(1, *edge_index.size())  # Add a batch dimension

    # Convert your data to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float32)

    # Train-test split without labels (unsupervised learning)
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)

    # Z-score normalization
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std


    model = GraphSAGEModel(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(x_train, edge_index)
        loss = F.mse_loss(out, x_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        reconstruction_test = model(x_test, edge_index)
        mse_test = F.mse_loss(reconstruction_test, x_test)
        print(f'Test Mean Squared Error: {mse_test:.4f}')

    # Read the test data
    test = pd.read_csv('0_gateway.csv')
    # Get the number of rows
    num_rows = test.shape[0]
    print(f"Number of rows in '0_gateway.csv': {num_rows}")


    total_detected_anomalies=0
    total_number_of_rootcauses=0
    list_of_indirect_independencies={}
    total_true_rootcauses=0
    avg_time=0
    rootcause=0 #rootcause value
    # num_test_data=5000
    k=0.2 #0.1,0.3
    num_test_data = int(790000 * k)
    test_data = test.sample(n=num_test_data, random_state=np.random.randint(0, num_test_data))
    test_data = test_data.iloc[:, 1:].values
    # num_processes = os.cpu_count()
    # num_threads=os.cpu_count()
    num_threads=1000
    num_rows=test_data

    args_list = [(model, edge_index, row_id, row, mean, std, num_services, num_features, 0.1, rootcause) for row_id, row
                 in enumerate(test_data)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(replace_anomalies, args_list))
            for idx, (anomaly_detection, true_rootcauses, local_number_of_rootcauses, List_of_candidates,elapsed_time) in enumerate(results):
                if anomaly_detection:
                    total_detected_anomalies+=1
                if true_rootcauses:
                    total_true_rootcauses +=1
                total_number_of_rootcauses+=local_number_of_rootcauses
                for item in List_of_candidates:
                    list_of_indirect_independencies[item] = list_of_indirect_independencies.get(item, 0) + 1
                #average execution time
                avg_time+=elapsed_time

    percentage_of_detected_anomaly=total_detected_anomalies/num_rows
    percenge_of_true_rootcause=total_true_rootcauses/num_rows
    avg_of_number_of_root_causes=total_number_of_rootcauses/num_rows
    avg=avg_time/num_rows

    # Print the results
    print("Percentage of detected anomalies:", percentage_of_detected_anomaly, "%")
    print("Percentage of true root causes:", percenge_of_true_rootcause, "%")
    print("Average number of root causes:", avg_of_number_of_root_causes)
    print("list_of_indirect_independencies:",set(list_of_indirect_independencies))
    print(f"Execution time: {avg} seconds")