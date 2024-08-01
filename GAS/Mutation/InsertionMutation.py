import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class InsertionMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            seq = individual.seq[:]
            pos1 = random.randint(0, len(seq) - 1)
            pos2 = random.randint(0, len(seq) - 1)
            gene = seq.pop(pos1)
            seq.insert(pos2, gene)
            return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
        return individual

