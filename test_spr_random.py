import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from IPython import embed
from tqdm import tqdm

from dataset.dataset_spr import SprInstructions

def test_manhattan(testloader, config, report_path=None):
    total, correct = 0.0, 0.0
    dist = []
    stop = []
    progress = tqdm(total = len(testloader))
    for i, data in enumerate(testloader):
        layouts, object_maps, inst, S1, S2, goals, labels = data
        # Send Tensors to GPU if available
        # if use_GPU:
        #     layouts = layouts.cuda()
        #     object_maps = object_maps.cuda()
        #     inst = inst.cuda()
        #     S1 = S1.cuda()
        #     S2 = S2.cuda()
        #     labels = labels.cuda()  
        # # Wrap to autograd.Variable
        # layouts, object_maps = Variable(layouts), Variable(object_maps)
        # S1, S2 = Variable(S1), Variable(S2)
        # inst = Variable(inst)
        d, st = simulate_rollout(layouts, object_maps, inst, S1, S2, goals, labels, config)
        dist.append(d)
        stop.append(st)
        progress.update(1)
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
    return


def simulate_rollout(layouts, object_maps, inst, S1, S2, goals, labels, config, max_steps=75):
    reached = False
    step = 0
    while (not reached) and (step < max_steps):
        # outputs, predictions, v, r, heatmap = net(layouts, object_maps, inst, S1, S2, config)
        # _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # predicted = predicted.data
        # action = int(predicted.view(-1)[0])
        # action = np.random.randint(0, 4)
        action = np.random.randint(0, 8)
        # embed()
        # exit()
        s1_new, s2_new = get_start_postion(action, int(S1), int(S2), reduced=False)
        # S1 = Variable(torch.from_numpy(np.array([s1_new]))).cuda()
        # S2 = Variable(torch.from_numpy(np.array([s2_new]))).cuda()
        S1 = s1_new
        S2 = s2_new
        reached = goal_reached(goals.view(-1), int(S1), int(S2))
        step += 1

    #calculate manhattan distance
    dist = calc_manhattan(goals.view(-1), s1_new, s2_new)
    return dist, step

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

if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='trained_models/spr_vin_10x10.pth', 
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
                        default=2, 
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
    testset = SprInstructions(config.datadir, mode=config.map_type, annotation='human', train=False, scale=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    test_manhattan(testloader, config)