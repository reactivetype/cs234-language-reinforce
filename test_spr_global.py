import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
# import visualization
import torch
from torch.autograd import Variable
from IPython import embed
from tqdm import tqdm
import pickle

from dataset.dataset_spr import SprInstructions

def test_manhattan(net, testloader, config, report_path=None, testset=None):
    # testset = SprInstructions(config.datadir, mode='local', annotation='human', train=False, scale=1)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    use_GPU = torch.cuda.is_available()
    # net = torch.load(config.weights)
    # if use_GPU:
    #     net = net.cuda()
    total, correct = 0.0, 0.0
    dist = []
    stop = []
    progress = tqdm(total = len(testloader))
    info_dict = initialize_recording_struct()
    for i, data in enumerate(testloader):
        layouts, object_maps, inst, S1, S2, goals, labels = data
        inst_text = testset.instructions[i]
        reward = testset.rewards[i]
        val = testset.values[i]
        info_dict = record_truth(info_dict, layouts, object_maps, inst_text, goals, reward, val)

        # Send Tensors to GPU if available
        if use_GPU:
            layouts = layouts.cuda()
            object_maps = object_maps.cuda()
            inst = inst.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            labels = labels.cuda()  
        # Wrap to autograd.Variable
        layouts, object_maps = Variable(layouts), Variable(object_maps)
        S1, S2 = Variable(S1), Variable(S2)
        inst = Variable(inst)
        # Forward pass
        # embed()
        # outputs, predictions = net(layouts, object_maps, inst, S1, S2, config)
        # # Select actions with max scores(logits)
        # _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # # Unwrap autograd.Variable to Tensor
        # predicted = predicted.data
        # # Compute test accuracy
        # correct += (predicted.view(-1) == labels).sum()
        # total += labels.size()[0]
        # embed()
        # break
        d, st, info_dict = simulate_rollout(net, layouts, object_maps, inst,
                                            S1, S2, goals, labels, config, info_dict)
        dist.append(d)
        stop.append(st)
        progress.update(1)
        # if i > 3:
        #     break
    # print('Test Accuracy: {:.2f}%'.format(100*(correct/total)))
    # print "Average Manhattan distance: %s" %(np.mean(dist))
    print "Manhattan Distance:"
    dist_report = get_stats(dist)
    print dist_report

    print "Stopping step:"
    steps_report = get_stats(stop)
    print steps_report

    if report_path:
        with open(report_path, 'w') as rid:
            rid.write('Manhattan Distance:\n')
            rid.write(dist_report+'\n')
            rid.write("Stopping step:\n")
            rid.write(steps_report+'\n')
        pkl_path = report_path.replace('.txt', '.pkl')
        with open(pkl_path, 'wb') as pid:
            pickle.dump(info_dict, pid)
    return


def simulate_rollout(net, layouts, object_maps, inst, S1, S2, goals, labels, config, info_dict, max_steps=75):
    reached = False
    step = 0
    info_dict = initialize_rollout_record(info_dict, S1, S2)
    while (not reached) and (step < max_steps):
        outputs, predictions, v, r, heatmap = net(layouts, object_maps, inst, S1, S2, config)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        predicted = predicted.data
        action = int(predicted.view(-1)[0])
        # embed()
        # exit()
        s1_new, s2_new = get_start_postion(action, int(S1), int(S2))
        next_state = (s1_new, s2_new)
        if step == 0:
            info_dict = record_prediction(info_dict, action, next_state, v, r)
        else:
            info_dict = record_action(info_dict, action, next_state)
        S1 = Variable(torch.from_numpy(np.array([s1_new]))).cuda()
        S2 = Variable(torch.from_numpy(np.array([s2_new]))).cuda()
        reached = goal_reached(goals.view(-1), int(S1), int(S2))
        step += 1

    #calculate manhattan distance
    dist = calc_manhattan(goals.view(-1), s1_new, s2_new)
    info_dict['prediction']['distance'].append(dist)
    return dist, step, info_dict

def get_start_postion(action, s1, s2, grid_size=10, reduced=True):
    if reduced:
        return get_start_postion_four(action, s1, s2, grid_size)
    else:
        return get_start_postion_eight(action, s1, s2, grid_size)

def get_start_postion_four(action, s1, s2, grid_size=10):
    """
        s2 is the inner dimension of the 2D array
               (s1-1, s2)/2
    (s1,s2-1)/3            (s1, s2+1)/1
               (s1+1, s2)/0 
    """
    if action in [3]:
        s2_new = s2 -1
    elif action in [1]:
        s2_new = s2+1
    else:
        s2_new = s2

    if action in [2]:
        s1_new = s1 -1
    elif action in [0]:
        s1_new = s1+1
    else:
        s1_new = s1

    #check for overflow
    of = False
    for s in [s1_new, s2_new]:
        if s<0 or s>grid_size-1:
            of = True
            break
    if of:
        s1_new, s2_new = s1, s2

    return s1_new, s2_new

def get_start_postion_eight(action, s1, s2, grid_size=10):
    """
    Action mapping:
       s2(-1  0   1)
    s1-1  4   5   3
       0  6       2 
       1  7   0   1
 
    S1: Y-axis
    S2: X-axis
    """
    #Refer to the layout above
    #Update for s2
    if action in [4, 6, 7]:
        s2_new = s2 -1
    elif action in [5, 0]:
        s2_new = s2
    else:
        s2_new = s2 + 1

    #update for s1
    if action in [4, 5, 3]:
        s1_new = s1 -1
    elif action in [6, 2]:
        s1_new = s1
    else:
        s1_new = s1 + 1

    #check for overflow
    of = False
    for s in [s1_new, s2_new]:
        if s<0 or s>grid_size-1:
            of = True
            break
    if of:
        s1_new, s2_new = s1, s2

    return s1_new, s2_new


def goal_reached(goal, s1, s2):
    """
    goals(s1, s2)
    """
    s1_dest = int(goal[0])
    s2_dest = int(goal[1])
    if s1_dest == s1 and s2_dest == s2:
        return True
    else:
        return False

def calc_manhattan(goal, s1, s2):
    """
    """
    s1_dest = int(goal[0])
    s2_dest = int(goal[1])
    dist = np.abs(s1_dest-s1) + np.abs(s2_dest-s2)
    return dist

def visualize(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:,0], states_xy[:,1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:,0], pred_traj[:,1], '-X', c='r', label='Predicted Path')
    ax.plot(states_xy[0,0], states_xy[0,1], '-o', label='Start')
    ax.plot(states_xy[-1,0], states_xy[-1,1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small') # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

def get_stats(arr):
    m = np.mean(arr)
    md = np.median(arr)
    mn = np.min(arr)
    mx = np.max(arr)
    std = np.std(arr)
    stats = "Mean:%s, Std:%s, Median:%s, Min-Max: %s,%s" %(m, std, md, mn, mx)
    return stats

def initialize_recording_struct():
    info_dict = {}
    #Ground Truth info
    info_dict['truth'] = {}
    info_dict['truth']['layout'] = []
    info_dict['truth']['object'] = []
    info_dict['truth']['instruction'] = []
    info_dict['truth']['reward'] = []
    info_dict['truth']['value'] = []
    info_dict['truth']['goal'] = []

    info_dict['prediction'] = {}
    info_dict['prediction']['start_pos'] = []
    info_dict['prediction']['action'] = []
    info_dict['prediction']['next_state'] = []
    info_dict['prediction']['rhat'] = []
    info_dict['prediction']['vhat'] = []
    info_dict['prediction']['distance'] = []
    return info_dict

def record_truth(info_dict, layout, objects, instruction, goal, reward, value):
    """
    Records ground truth
    """
    info_dict['truth']['layout'].append(layout)
    info_dict['truth']['object'].append(objects)
    info_dict['truth']['instruction'].append(instruction)
    info_dict['truth']['reward'].append(reward)
    info_dict['truth']['value'].append(value)
    info_dict['truth']['goal'].append(goal)
    return info_dict

def initialize_rollout_record(info_dict, S1, S2):
    start_pos = (int(S2), int(S1))
    info_dict['prediction']['start_pos'].append(start_pos)
    info_dict['prediction']['action'].append([])
    info_dict['prediction']['next_state'].append([])
    # info_dict['prediction']['rhat'].append([])
    # info_dict['prediction']['vhat'].append([])
    return info_dict

def record_prediction(info_dict, action, next_state, vhat, rhat):
    info_dict['prediction']['action'][-1].append(action)
    info_dict['prediction']['next_state'][-1].append(next_state)
    info_dict['prediction']['rhat'].append(rhat.cpu().squeeze().data.numpy())
    info_dict['prediction']['vhat'].append(vhat.cpu().squeeze().data.numpy())
    return info_dict

def record_action(info_dict, action, next_state):
    info_dict['prediction']['action'][-1].append(action)
    info_dict['prediction']['next_state'][-1].append(next_state)
    return info_dict

if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='', 
                        help='Path to trained weights')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--imsize', 
                        type=int, 
                        default=10, 
                        help='Size of image')
    parser.add_argument('--k', 
                        type=int, 
                        default=10, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=3, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=5, 
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--datadir', 
                        type=str, 
                        default='spr_data/', 
                        help='Path to data file')
    parser.add_argument('--map_type', 
                        type=str, 
                        default="local", 
                        help='Global or local maps')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='Batch size')
    config = parser.parse_args()
    # Compute Paths generated by network and plot
    testset = SprInstructions(config.datadir, mode=config.map_type, annotation='human', train=True, scale=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    # embed()
    net = torch.load(config.weights)
    #ste batch size to one
    # net.positions_batch = net.positions.repeat(1,1,1,1)
    test_manhattan(net, testloader, config)