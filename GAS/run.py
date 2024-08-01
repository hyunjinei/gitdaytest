import os
import sys
import random
import copy
import csv
import datetime
from multiprocessing import Pool, Value, Array, Manager, Lock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from GAS.GA import GAEngine
# Crossover
from GAS.Crossover.PMX import PMXCrossover
from GAS.Crossover.CX import CXCrossover
from GAS.Crossover.LOX import LOXCrossover
from GAS.Crossover.OrderBasedCrossover import OBC
from GAS.Crossover.PositionBasedCrossover import PositionBasedCrossover
from GAS.Crossover.SXX import SXX
from GAS.Crossover.PSX import PSXCrossover
from GAS.Crossover.OrderCrossover import OrderCrossover
from GAS.Crossover.POXCrossover import POXCrossover

# Mutation
from GAS.Mutation.GeneralMutation import GeneralMutation
from GAS.Mutation.DisplacementMutation import DisplacementMutation
from GAS.Mutation.InsertionMutation import InsertionMutation
from GAS.Mutation.ReciprocalExchangeMutation import ReciprocalExchangeMutation
from GAS.Mutation.ShiftMutation import ShiftMutation
from GAS.Mutation.InversionMutation import InversionMutation
from GAS.Mutation.SwapMutation import SwapMutation
# from GAS.Mutation.IntelligentMutation import IntelligentMutation

# Selection
from GAS.Selection.RouletteSelection import RouletteSelection
from GAS.Selection.SeedSelection import SeedSelection
from GAS.Selection.TournamentSelection import TournamentSelection
# from GAS.Selection.TruncationSelection import TruncationSelection

# Local Search
from Local_Search.HillClimbing import HillClimbing
from Local_Search.TabuSearch import TabuSearch
from Local_Search.SimulatedAnnealing import SimulatedAnnealing
from Local_Search.GifflerThompson_LS import GifflerThompson_LS

# Meta Heuristic
from Meta.PSO import PSO  # pso를 추가합니다

# 선택 mutation 
from GAS.Mutation.SelectiveMutation import SelectiveMutation

from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log  # 수정된 부분


'''
txt : TARGET_MAKESPAN, Jobs, Machines

la01: 666  10, 5/  la11: 1222  20, 5
la02: 655  10, 5/  la12: 1039  20, 5
la03: 597  10, 5/  la13: 1150  20, 5
la04: 590  10, 5/  la14: 1292  20, 5
la05: 593  10, 5/  la15: 1207  20, 5
la06: 926  15, 5/  la16: 945   10, 10
la07: 890  15, 5/  la17: 784   10, 10
la08: 863  15, 5/  la18: 848   10, 10
la09: 951  15, 5/  la19: 842   10, 10
la10: 958  15, 5/  la20: 902   10, 10

ta21: 1642 20 20/  ta51: 2760 50 15
ta22: 1561 1600 20 20/  ta52: 2756 50 15
ta31: 1764 30 15/  ta61: 2868 50 20
ta32: 1774 1784 30 15/  ta62: 2869 50 20
ta41: 1906 2005 30 20/  ta71: 5464 100 20
ta42: 1884 1937 30 20/  ta72: 5181 100 20

abz5 = 1234  10, 10
ft20 = 1165
'''

TARGET_MAKESPAN = 1642  # 목표 Makespan
MIGRATION_FREQUENCY = 300  # Migration frequency 설정
random_seed = 42  # Population 초기화시 일정하게 만들기 위함. None을 넣으면 아예 랜덤 생성(GA들끼리 같지않음)



def run_ga_engine(args):
    ga_engine, index, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, events = args
    try:
        best, best_crossover, best_mutation, all_generations, execution_time, best_time = ga_engine.evolve(index, sync_generation, sync_lock, events)
        
        # GA 엔진 상태 출력
        print(f"GA{index+1} 상태:")
        print(f"Population: {ga_engine.population is not None}")
        print(f"Best Individual: {best is not None}")
        print(f"Current Generation: {sync_generation[index]}")
        
        if best is None:
            return None

        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga_engine.selection.__class__.__name__
        local_search_names = [ls.__class__.__name__ for ls in ga_engine.local_search]
        local_search_name = "_".join(local_search_names)
        pso_name = ga_engine.pso.__class__.__name__ if ga_engine.pso else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_path = os.path.join(result_txt_path, f'log_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
        generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')

        if best is not None and hasattr(best, 'monitor'):
            best.monitor.save_event_tracer(log_path)
            ga_engine.config.filename['log'] = log_path
            generated_log_df = generate_machine_log(ga_engine.config)
            generated_log_df.to_csv(machine_log_path, index=False)
            ga_engine.save_csv(all_generations, execution_time, generations_path)
        else:
            print("No valid best individual or monitor to save the event tracer.")

        return best, best_crossover, best_mutation, all_generations, execution_time, best_time, index
    except Exception as e:
        print(f"Exception in GA {index+1}: {e}")
        return None


def main():
    print("Starting main function...")  # 디버그 출력 추가
    # initialization_mode = input("Select Initialization GA mode (1: basic, 2: MIO, 3: GifflerThompson): ")
    # print(f"Selected Initialization GA mode: {initialization_mode}")

    island_mode = int(input("Select Island-Parallel GA mode (1: Independent, 2: Sequential Migration, 3: Random Migration): "))
    print(f"Selected Island-Parallel GA mode: {island_mode}")

    file = 'ta21.txt'
    print(f"Loading dataset from {file}...")  # 디버그 출력 추가
    dataset = Dataset(file)

    base_config = Run_Config(n_job=20, n_machine=20, n_op=400, population_size=500, generations=100, 
                             print_console=False, save_log=True, save_machinelog=True, 
                             show_gantt=False, save_gantt=True, show_gui=False,
                             trace_object='Process4', title='Gantt Chart for JSSP',
                             tabu_search_iterations=10, hill_climbing_iterations=30, simulated_annealing_iterations=50)
    
    print("Base config created...")  # 디버그 출력 추가

    base_config.dataset_filename = file  # dataset 파일명 설정
    base_config.target_makespan = TARGET_MAKESPAN  # 목표 Makespan
    base_config.island_mode = island_mode  # Add this line to set island_mode

    result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    result_txt_path = os.path.join(result_path, 'result_txt')
    result_gantt_path = os.path.join(result_path, 'result_Gantt')
    ga_generations_path = os.path.join(result_path, 'ga_generations')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_txt_path):
        os.makedirs(result_txt_path)
    if not os.path.exists(result_gantt_path):
        os.makedirs(result_gantt_path)
    if not os.path.exists(ga_generations_path):
        os.makedirs(ga_generations_path)

    print("Result directories checked/created...")  # 디버그 출력 추가

    '''
    crossovers
    [OrderCrossover, PMXCrossover, LOXCrossover, OBC, 
    PositionBasedCrossover, SXX,PSXCrossover,POXCrossover]  # Crossover 리스트
    '''

    '''
    mutations 
    [GeneralMutation, DisplacementMutation, InsertionMutation, 
    ReciprocalExchangeMutation,ShiftMutation, InversionMutation, SwapMutation,DiverseSwapMutation]
    '''

    '''
    selection 
    [TournamentSelection(), SeedSelection(), RouletteSelection(),DiverseTournamentSelection()]
    '''

    '''
    Local Search
    [HillClimbing(), TabuSearch(), SimulatedAnnealing(), GifflerThompson()] # Local Search 리스트
    GifflerThompson(priority_rule='SPT') -> SPT, LPT, MWR, LWR, MOR, LOR, EDD, FCFS, RANDOM
    '''

    '''
    Meta Heuristic
    ['pso': PSO(num_particles=10, num_iterations=50)], 'pso': None  # PSO 추가
    '''

    custom_settings = [
        # {'crossover': POXCrossover, 'pc': 0.7, 'mutation': IntelligentMutation, 'pm': 0.05, 'selection': TruncationSelection(),'elite_TS': 0.2, 'local_search': [], 'pso':  None, 'selective_mutation': SelectiveMutation(pm_high=0.5, pm_low=0.01, rank_divide=0.4)},
        {'crossover': OrderCrossover, 'pc': 0.8, 'mutation': InsertionMutation, 'pm': 0.1, 'selection': TournamentSelection(), 'local_search': [], 'pso':  None, 'selective_mutation': SelectiveMutation(pm_high=0.7, pm_low=0.4, rank_divide=0.05)},
        {'crossover': OrderCrossover, 'pc': 0.8, 'mutation': InsertionMutation, 'pm': 0.1, 'selection': TournamentSelection(), 'local_search': [], 'pso':  None, 'selective_mutation': SelectiveMutation(pm_high=0.8, pm_low=0.1, rank_divide=0.05)},
    ]

# (pm_high=0.5, pm_low=0.01, rank_divide=0.4) 기존

    ga_engines = []
    for i, setting in enumerate(custom_settings):
        crossover_class = setting['crossover']
        mutation_class = setting['mutation']
        selection_instance = setting['selection']
        # selection_instance.elite_TS = setting['elite_TS']  # 여기에 설정합니다.
        local_search_methods = setting['local_search']
        pso_class = setting.get('pso')
        selective_mutation_instance = setting['selective_mutation']
        pc = setting['pc']
        pm = setting['pm']

        initialization_mode = input(f"Select Initialization GA mode for GA{i+1} (1: basic, 2: MIO, 3: GifflerThompson): ")
        print(f"Selected Initialization GA mode for GA{i+1}: {initialization_mode}")

        config = copy.deepcopy(base_config)
        config.filename['log'] = os.path.join(result_txt_path, f'GA{i+1}_{config.now}.csv')
        config.filename['machine'] = os.path.join(result_txt_path, f'GA{i+1}_{config.now}_machine.csv')
        config.filename['gantt'] = os.path.join(result_gantt_path, f'GA{i+1}_{config.now}.png')
        config.filename['csv'] = os.path.join(ga_generations_path, f'GA{i+1}_{config.now}.csv')

        config.ga_index = i + 1

        crossover = crossover_class(pc=pc)
        mutation = mutation_class(pm=pm)
        selection = selection_instance
        pso = pso_class if pso_class else None
        local_search = local_search_methods
        local_search_frequency = 40
        selective_mutation_frequency = 20
        selective_mutation = selective_mutation_instance

        if initialization_mode == '1':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '2':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='2', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '3':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='3', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        
        ga_engines.append(ga_engine)

        print(f"Initialized GAEngine {i+1}")  # 디버그 출력 추가

    best_individuals = [None] * len(ga_engines)
    stop_evolution = Manager().Value('i', 0)
    elite_population = Manager().list([None] * len(ga_engines))

    manager = Manager()
    sync_generation = manager.list([0] * len(ga_engines))
    sync_lock = manager.Lock()

    if island_mode in ['2', '3']:
        events = [manager.Event() for _ in range(len(ga_engines))]

    with Pool() as pool:
        while True:
            if island_mode in ['2', '3']:
                args = [(ga_engines[i], i, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, events) for i in range(len(ga_engines))]
            else:
                args = [(ga_engines[i], i, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, None) for i in range(len(ga_engines))]
            
            results = pool.map(run_ga_engine, args)

            all_completed = True
            for result in results:
                if result is not None:
                    best, best_crossover, best_mutation, all_generations, execution_time, best_time, index = result
                    best_individuals[index] = (best, best_crossover, best_mutation, execution_time, best_time, all_generations)
                    elite_population[index] = best
                    crossover_name = best_crossover.__class__.__name__
                    mutation_name = best_mutation.__class__.__name__
                    selection_name = ga_engines[index].selection.__class__.__name__
                    local_search_names = [ls.__class__.__name__ for ls in ga_engines[index].local_search]
                    local_search_name = "_".join(local_search_names)
                    pso_name = ga_engines[index].pso.__class__.__name__ if ga_engines[index].pso else 'None'
                    pc = best_crossover.pc
                    pm = best_mutation.pm
                    log_path = os.path.join(result_txt_path, f'log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')

                    if best is not None and hasattr(best, 'monitor'):
                        best.monitor.save_event_tracer(log_path)
                        ga_engines[index].config.filename['log'] = log_path
                        generated_log_df = generate_machine_log(ga_engines[index].config)
                        generated_log_df.to_csv(machine_log_path, index=False)
                        ga_engines[index].save_csv(all_generations, execution_time, generations_path)
                    else:
                        print("No valid best individual or monitor to save the event tracer.")

                    if best.makespan <= TARGET_MAKESPAN:
                        stop_evolution.value = 1
                        print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                        break

                    if os.path.exists(log_path) and os.path.exists(machine_log_path) and os.path.exists(generations_path):
                        stop_evolution.value = 1
                        print(f"Stopping as all files for GA{index+1} are generated.")
                        break
                else:
                    all_completed = False

            if stop_evolution.value or all_completed:
                break

    for i, result in enumerate(best_individuals):
        if result is not None:
            best, best_crossover, best_mutation, execution_time, best_time, all_generations = result
            crossover_name = best_crossover.__class__.__name__
            mutation_name = best_mutation.__class__.__name__
            selection_name = ga_engines[i].selection.__class__.__name__
            local_search_names = [ls.__class__.__name__ for ls in ga_engines[i].local_search]
            local_search_name = "_".join(local_search_names)
            pso_name = ga_engines[i].pso.__class__.__name__ if ga_engines[i].pso else 'None'
            pc = best_crossover.pc
            pm = best_mutation.pm
            print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and selection: {selection_name} and Local Search: {local_search_name} and pso: {pso_name}, Time taken: {execution_time:.2f} seconds, First best time: {best_time:.2f} seconds")
            machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
            gantt_path = os.path.join(result_gantt_path, f'gantt_chart_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.png')
            if os.path.exists(machine_log_path):
                machine_log = pd.read_csv(machine_log_path)
                ga_engines[i].config.filename['gantt'] = gantt_path
                Gantt(machine_log, ga_engines[i].config, best.makespan)
            else:
                print(f"Warning: {machine_log_path} does not exist.")

if __name__ == "__main__":
    main()
