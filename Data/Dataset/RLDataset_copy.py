# RLDataset.py
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RLDataset:
    def __init__(self, filename):
        self.name, _ = os.path.splitext(filename)
        file_path = os.path.join(os.getcwd(), filename)
        
        with open(file_path, 'r') as file:
            first_line = file.readline()

        self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
        self.n_op = self.n_job * self.n_machine

        self.op_data = []
        data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", skiprows=[0], header=None)

        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                self.op_data[i].append((data.iloc[self.n_job + i, j] - 1, data.iloc[i, j]))

        print(f"Loaded data from {filename}")
        print(f"Number of jobs: {self.n_job}, Number of machines: {self.n_machine}")

def load_all_datasets(filenames):
    datasets = []
    for filename in filenames:
        dataset_loader = RLDataset(filename)
        datasets.append(dataset_loader)
    return datasets