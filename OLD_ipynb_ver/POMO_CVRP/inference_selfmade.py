from HYPER_PARAMS import *  # NOTE : You much edit HYPER_PARAMS to match the model you are loading
from TORCH_OBJECTS import *
import numpy as np
import time
import os
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)

from source.cvrp import GROUP_ENVIRONMENT
import source.MODEL__Actor.grouped_actors as A_Module
from source.utilities import Average_Meter, augment_xy_data_by_8_fold, Get_Logger
import matplotlib.pyplot as plt



def plot(fig_folder_name, all_action, node_data, demand_data, episode):
    cmap = plt.get_cmap("tab20")
    for i in range(node_data.shape[0]):
        plt.figure(figsize=(10,10))
        part_node_data = node_data[i].cpu()
        part_demand_data = demand_data[i].cpu()
        part_action = all_action[i]
        color = 0
        all_dist_list = []
        part_dist = 0
        s = np.concatenate([[1], np.array(part_demand_data[:,0]) *  1000])
        plt.scatter(part_node_data[:, 0], part_node_data[:, 1], s= s)
        # 追加した部分
        if set(np.array(part_action.cpu().detach())) != {i for i in range(101)}:
            print("unvisited node is detected!")
            breakpoint()

        visited = set()
        all_visited_list = []
        visited_list = list()
        for (u, v) in zip(part_action, part_action[1:]):
            visited_list.append(int(u))
            plt.plot([part_node_data[u][0], part_node_data[v][0]], [part_node_data[u][1], part_node_data[v][1]], color=cmap(color))
            part_dist += float(torch.norm(part_node_data[u] - part_node_data[v]))
            if v == 0:
                all_dist_list.append(round(part_dist, 3))
                visited_list.append(int(v))
                all_visited_list.append(visited_list)
                visited_list = list()
                part_dist = 0
                color += 1
                color = color % 20
            # if visited == {i for i in range(101)}:
            #     all_dist_list.append(round(part_dist, 3))
            #     break
        epsilon = 0.01
        for v in all_visited_list:
            print(list(v))
        plt.xlim([0-epsilon, 1+epsilon])
        plt.ylim([0-epsilon, 1+epsilon])
        plt.title(f"{round(float(sum(part_demand_data)), 3)} / {len(all_dist_list) - 1} / {"-".join(list(map(str, all_dist_list)))}")
        filename = f"{fig_folder_name}/output{str(i).zfill(4)}_{str(episode).zfill(4)}.png"
        print(filename)
        plt.savefig(filename)
        plt.close()


def main():
    SAVE_FOLDER_NAME = "INFERENCE_00"
    print(SAVE_FOLDER_NAME)

    fig_folder_name = "fig"
    os.makedirs(fig_folder_name, exist_ok=True)


    # Make Log File
    logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
    print(result_folder_path)

    # Load Model
    grouped_actor = A_Module.ACTOR().to(device)

    actor_model_save_path = './result/Saved_CVRP100_Model/ACTOR_state_dic.pt'
    grouped_actor.load_state_dict(torch.load(actor_model_save_path, map_location="cuda:0"))
    grouped_actor.eval()

    logger.info('==============================================================================')
    logger.info('==============================================================================')
    log_str = '  <<< MODEL: {:s} >>>'.format(actor_model_save_path)
    logger.info(log_str)


    if PROBLEM_SIZE == 20:
        demand_scaler = 30
    elif PROBLEM_SIZE == 50:
        demand_scaler = 40
    elif PROBLEM_SIZE == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError


    LOG_PERIOD_SEC = 10
    # DATASET_SIZE = 100 *1000
    DATASET_SIZE = 4

    # TEST_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 4
    eval_dist_AM_0 = Average_Meter()


    logger.info('===================================================================')
    log_str = 'Single Trajectory'
    logger.info(log_str)

    timer_start = time.time()
    logger_start = time.time()


    episode = 0
    while True:
        node_data = Tensor(np.random.rand(TEST_BATCH_SIZE, PROBLEM_SIZE+1, 2))
        # demand_data = Tensor(np.random.randint(1, 10, TEST_BATCH_SIZE*PROBLEM_SIZE) / demand_scaler)
        demand_data = Tensor(np.ones(TEST_BATCH_SIZE*PROBLEM_SIZE) / 25)
        
        depot_xy = node_data[:, [0], :]
        node_xy = node_data[:, 1:, :]
        node_demand = demand_data.reshape(TEST_BATCH_SIZE, PROBLEM_SIZE, 1)
        # node_demand *= 0.43
        # depot_xy.shape = (batch, 1, 2)
        # node_xy.shape = (batch, problem, 2)
        # node_demand.shape = (batch, problem, 1)

        batch_s = TEST_BATCH_SIZE
        episode = episode + batch_s
        
        

        with torch.no_grad():

            env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand)
            group_s = 1
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
            all_action = first_action.clone()
            group_state, reward, done = env.step(first_action)

            # Second Move is given
            second_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            all_action = torch.cat([all_action, second_action], axis=1)
            group_state, reward, done = env.step(second_action)

            # batch内の全jobが終了したら止まりそう
            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem+1)
                action = action_probs.argmax(dim=2)
                all_action = torch.cat([all_action, action], axis=1)
                # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
        plot(fig_folder_name, all_action, node_data, node_demand, episode)

        eval_dist_AM_0.push(-reward[:, 0])  # reward was given as negative dist



        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode >= DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:07d}({:5.1f}%)  T:{:s}  avg.dist:{:f}'\
                .format(episode, episode/DATASET_SIZE*100,
                        timestr, eval_dist_AM_0.peek())
            logger.info(log_str)
            logger_start = time.time()

        if episode >= DATASET_SIZE:
            break
        
        
    logger.info('---------------------------------------------------')
    logger.info('average = {}'.format(eval_dist_AM_0.result()))
    logger.info('---------------------------------------------------')
    logger.info('---------------------------------------------------')






if __name__ == "__main__":
    main()