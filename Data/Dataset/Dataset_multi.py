import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from GAS_FJSP.Individual import Individual

class Dataset:
    def __init__(self, filename):
        self.name, _ = os.path.splitext(filename)
        self.path = 'Data\\Dataset\\'
        if __name__ == "__main__":
            file_path = os.path.join(os.getcwd(), filename)
        else:
            file_path = os.path.join(os.path.dirname(__file__), filename)

        with open(file_path, 'r') as file:
            first_line = file.readline()

        self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
        self.n_op = self.n_job * self.n_machine

        self.op_data = []
        data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", skiprows=[0], header=None)
        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                machine_id = data.iloc[self.n_job + i, j] - 1
                process_time = data.iloc[i, j]
                part_name = f"Part{i}_{j}"
                self.op_data[i].append((machine_id, process_time, part_name))

        self.n_solution = 0
