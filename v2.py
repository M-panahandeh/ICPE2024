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
import glob
from scipy.stats import lognorm

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

        # Calculate mean squared error (MSE) between input and reconstruction
        mse = F.mse_loss(reconstruction, input_row).item()

        # Check if MSE is below the threshold
        in_distribution = mse < threshold
        dev=(mse - threshold)

    return in_distribution, dev
##############################################################

# Function to replace anomaly columns with mean values
def replace_anomalies(args):
    # Record the start time
    start_time = time.time()

    model, edge_index, row_id,input_row, mean, std, num_services, num_features, threshold, rootcause,critical_path = args

    anomaly_detection=False
    true_rootcauses = False
    local_number_of_rootcauses=0
    List_of_candidates =[]
    deviation_list = []

    # anomalyDetection
    output,dev=is_in_distribution(model, edge_index, input_row, mean, std, num_services, num_features, threshold)

    if not output:
        anomaly_detection=True
       #culprit:
        for i in critical_path:
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
            is_in_dist, dev = is_in_distribution(model, edge_index, new_row, mean, std, num_services, num_features, threshold)


            if  is_in_dist:
                local_number_of_rootcauses+=1
                List_of_candidates.append(i)
                deviation_list.append((i, dev))
                if i == rootcause:  # true root cause
                    true_rootcauses=True

        deviation_list.sort(key=lambda x: x[1])
        for i, deviation in deviation_list:
            print("Index:", i, "Deviation:", deviation)
    print(f"Done processing {row_id}th row")
    # Acquire the lock before updating shared variables
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    return anomaly_detection,true_rootcauses,local_number_of_rootcauses,List_of_candidates,elapsed_time
def convert_to_paths(relationships):
    paths = []
    current_path = []

    for edge in relationships:
        if not current_path:
            current_path.append(edge[0])
        if current_path[-1] == edge[0]:
            current_path.append(edge[1])
        else:
            paths.append(current_path)
            current_path = [edge[0], edge[1]]

    if current_path:
        paths.append(current_path)

    return paths
####################################################################


if __name__ == '__main__':

    # Load the data
    data = pd.read_csv('no-interference.csv')
    num_train_data=len(data)
    data = data.sample(n=num_train_data, random_state=random.randint(0, num_train_data))


    # Number of services (nodes) and size of the feature vector for each service
    num_services = len(data.columns) - 1
    num_features = 1  # number of features for each service

    # Extract features and labels
    features = data.iloc[:, 1:].values
    features = np.array(features)
    features = features.reshape((data.shape[0], num_services,
                                 num_features))

    # Construct the directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_services))

    # Define the relationships based on your data
    relationships = [
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 7),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 8),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 9),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 10),
        (0, 1), (1, 2), (2, 4), (4, 6), (6, 11),
        (0, 1), (1, 2), (2, 5),
        (0, 1), (1, 3)
        # (0,1),(1, 5),(5, 13),
        # (0, 1), (1, 5), (5, 14),
        # (0, 2), (2, 6), (6, 22),
        # (0, 2), (2, 6), (6, 23),
        # (0, 3), (3, 7), (7, 11),(11,19),
        # (0, 3), (3, 8), (8, 12), (12, 28),
        # (0, 3), (3, 7), (7, 11), (11, 20),
        # (0, 3), (3, 8), (8, 12), (12, 29),
        # (0, 3), (3, 9), (9, 16),
        # (0, 3), (3, 9), (9, 17),
        # (0, 4), (4, 10), (10, 25),
        # (0, 4), (4, 10), (10, 26),
        # (0, 1), (1, 5), (5, 15),
        # (0, 2), (2, 6), (6, 24),
        # (0, 3), (3, 7), (7, 11),(11,21),
        # (0, 3), (3, 8), (8, 12), (12, 30),
        # (0, 3), (3, 9), (9, 18),
        # (0, 4), (4, 10), (10, 27)
    ]

    # Add edges to the graph based on relationships
    graph.add_edges_from(relationships)

    # Get edge indices as a PyTorch tensor
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_index = edge_index.view(1, *edge_index.size())  # Add a batch dimension

    # Convert your data to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float32)

    # Train-test(validation) split without labels (unsupervised learning)
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
        loss = F.mse_loss(out, x_train)  # Use mean squared error as a reconstruction loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        reconstruction_test = model(x_test, edge_index)
        mse_test = F.mse_loss(reconstruction_test, x_test)
        print(f'Test Mean Squared Error: {mse_test:.4f}')

    #find mean and std for traindata in clusters
    data = data.iloc[:,1:]
    paths=convert_to_paths(relationships)
    # Initialize a column for the max path
    data['critical_path'] = ""

    for index, row in data.iterrows():
        max_path = max(paths, key=lambda path: row[path].sum())
        data.at[index, 'critical_path'] = tuple(max_path)

    # Cluster by critical path
    grouped_data = data.groupby('critical_path')

    grouped_mean = {}
    grouped_std = {}
    for group, group_df in grouped_data:
        # Compute mean and std for each column within the group
        group_df = group_df.drop('critical_path', axis=1)
        mean_values = group_df.mean()
        std_values = group_df.std()
        # Store mean and std values in dictionaries
        grouped_mean[group] = mean_values
        grouped_std[group] = std_values
    # print(type(grouped_mean),grouped_std)


    # Read the test data ( all labelled files with Culprit)
    directory_path = r'D:\MYDESK\MyPhd\--Thesis--\DataChallenge\tracing-data\tracing-data\ticket-booking'
    file_pattern = directory_path + '/*.csv'
    # List all CSV files
    csv_files = glob.glob(file_pattern)
    with open('results-social.txt', 'a') as f:
     for file in csv_files:
        if 'no-interference' not in file:
            test = pd.read_csv(file)
            num_rows = test.shape[0]

            num_test_data=int(790000*0.2)
            test_data = test.sample(n=num_test_data, random_state=np.random.randint(0, num_test_data))
            test_data = test_data.iloc[:, 1:].values
            total_detected_anomalies=0
            total_number_of_rootcauses=0
            list_of_indirect_independencies={}
            total_true_rootcauses=0
            avg_time=0
            rootcause = int(os.path.splitext(os.path.basename(file))[0].split('_')[0])
            top_3_count = 0
            top_5_count = 0
            exam_score_sum = 0
            exam_score_count=0

            # num_threads=os.cpu_count()
            num_threads=10000
            args_list = [(model, edge_index, row_id, row, torch.tensor([grouped_mean[tuple(max(paths, key=lambda path: row[path].sum()))]][0].values, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor([grouped_std[tuple(max(paths, key=lambda path: row[path].sum()))]][0].values, dtype=torch.float32).unsqueeze(1),
                                  num_services, num_features, 0.1, rootcause,max(paths, key=lambda path: row[path].sum()))
                                 for row_id, row in enumerate(test_data)]

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
                                for index, item in enumerate(List_of_candidates):
                                    if index <= 2 and item == rootcause:
                                        top_3_count += 1
                                    if index <= 4 and item == rootcause:
                                        top_5_count += 1
                                    # Update exam score mean if item > 5
                                    if index > 5:
                                        exam_score_sum += index
                                        exam_score_count+1
                                #average execution time
                                avg_time+=elapsed_time

            percentage_of_detected_anomaly=total_detected_anomalies/num_test_data
            percenge_of_true_rootcause=total_true_rootcauses/num_test_data
            percenge_of_top3=top_3_count/num_test_data
            percenge_of_top5=top_5_count/num_test_data
            if exam_score_count>0:
                exam_score=exam_score_sum/exam_score_count
            else:
                  exam_score=0
            avg_of_number_of_root_causes=total_number_of_rootcauses/num_test_data
            avg=avg_time/num_test_data
            with open('results-social.txt', 'a') as f:
                # Print and write to file
                print("Percentage of detected anomalies:", percentage_of_detected_anomaly, "%")
                f.write(f"Percentage of detected anomalies: {percentage_of_detected_anomaly}%\n")
                print("Percentage of true root causes(ToP1):", percenge_of_true_rootcause, "%")
                f.write(f"Percentage of true root causes(ToP1): {percenge_of_true_rootcause}%\n")
                print("Percentage of true root causes(ToP3):", percenge_of_top3, "%")
                f.write(f"Percentage of true root causes(ToP3): {percenge_of_top3}%\n")
                print("Percentage of true root causes(ToP5):", percenge_of_top5, "%")
                f.write(f"Percentage of true root causes(ToP5): {percenge_of_top5}%\n")
                print("exam-score:", exam_score, "%")
                f.write(f"exam-score: {exam_score}%\n")
                print("Average number of root causes:", avg_of_number_of_root_causes)
                f.write(f"Average number of root causes: {avg_of_number_of_root_causes}\n")
                for item, count in list_of_indirect_independencies.items():
                    print("List of indirect independencies:", item, ":", count)
                    f.write("List of indirect independencies: " + str(item) + ": " + str(count) + "\n")
                print(f"Execution time: {avg} seconds")
                f.write(f"Execution time: {avg} seconds\n")
                print("----------------------------------")
                f.write("----------------------------------\n")
