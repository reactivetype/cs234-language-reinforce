import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
from IPython import embed
import environment

def load_pickle(p):
    with open(p, 'rb') as pid:
        d = pickle.load(pid)
    return d

def visualize_prediction(pkl_path, save_dir):
    dat = load_pickle(pkl_path)
    
    #fetch ground truth
    instructions = dat['truth']['instruction']
    layouts = np.array([i.cpu().squeeze().numpy() for i in dat['truth']['layout']])
    objects = np.array([i.cpu().squeeze().numpy() for i in dat['truth']['object']])
    l = layouts.reshape(len(layouts), 1, 10, 10)
    o = objects.reshape(len(objects), 1, 10, 10)
    l_o = np.concatenate((l, o), axis=1)
    print "Composite Shape {}".format(l_o.shape)
    
    ipath = os.path.join(save_dir, 'instructions.txt')
    with open(ipath, 'w') as instid:
        for i in instructions:
            instid.write(i+'\n')

    reward = dat['truth']['reward']
    val = dat['truth']['value']
    vhat = dat['prediction']['vhat']
    rhat = dat['prediction']['rhat']
    actions = dat['prediction']['action']
    val_summary, val_dist = calculate_peak_value_dist(val, vhat)
    get_success_rate(actions, 20)
    # same_inst_diff_maps(instructions, val)
    # # same_map_diff_inst(instructions, l_o)
    
    # # standalone plots
    plot_grid(val, save_dir, 'val', instructions)
    plot_grid(vhat, save_dir, 'vhat_vin', instructions)

    # #joint plots
    # joint_plot(val, vhat, save_dir, 'joint_vin', instructions, val_dist)
    # joint_plot(reward, rhat, save_dir, 'reward_vin', instructions, val_dist)
    return

def get_success_rate(actions, max_steps=25):
    a = np.array([len(i) - max_steps for i in actions])
    cnt = len(actions)
    print "Finding sucess rate for {} trials".format(cnt)
    s = np.count_nonzero(a<0)
    print "Sucess rate {}".format(s*100.0/cnt)
    return

def calculate_peak_value_dist(val, vhat):
    dist = []
    assert len(val) == len(vhat), 'Exapcted same no. of value maps'
    for i in range(len(val)):
        v = val[i].squeeze()
        vh = vhat[i].squeeze()
        g1 = np.unravel_index(v.argmax(), v.shape)
        g2 = np.unravel_index(vh.argmax(), vh.shape)
        d = calc_manhattan(g1, g2)
        dist.append(d)
    summ = get_statistics(dist, 'Peak Value Distance')
    return summ, dist

def calc_manhattan(g1, g2):
    dist = np.abs(g1[0]-g2[0]) + np.abs(g1[1] - g2[1])
    return dist

def get_statistics(a, info=None):
    p = "{} - Mean: {}, Median: {}, Std: {}".format(info, np.mean(a),
                                           np.median(a), np.std(a))
    print p
    return p

def plot_grid(val, save_dir, header_name, inst, colorbar=False):
    for i, (v, g) in enumerate(zip(val, inst)):
        plt.clf()
        f = plt.figure()
        p = plt.pcolor(v.squeeze(), cmap=cm.jet)
        # plt.title(g)
        p.axes.invert_yaxis()
        if colorbar: f.colorbar(p)
        save_path = os.path.join(save_dir, header_name+'_'+str(i)+'.png')
        plt.axis('off')
        plt.savefig(save_path)
    return

def joint_plot(val, vhat, save_dir, header_name, inst, dist=None, colorbar=True):
    for i, (v, vh, g) in enumerate(zip(val, vhat, inst)):
        plt.clf()
        fig, (ax0,ax1) = plt.subplots(1,2)
        h0 = ax0.pcolor(v.squeeze(), cmap=cm.jet)
        ax0.set_title('Ground Truth')
        ax0.set_aspect(1)
        h1 = ax1.pcolor(vh.squeeze(), cmap=cm.jet)
        ax1.set_title('VIN Prediction')
        ax0.invert_yaxis()
        ax1.invert_yaxis()
        ax1.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            labelleft='off')
        if colorbar:
            fig.colorbar(h0, ax=ax0, fraction=0.046, pad=0.04)
            fig.colorbar(h1, ax=ax1, fraction=0.046, pad=0.04)
        ax0.set_aspect(1)
        ax1.set_aspect(1)
        if dist:
            fig.suptitle(g + 'dist: %s' %dist[i])
        save_path = os.path.join(save_dir, header_name+'_'+str(i)+'.png')
        plt.savefig(save_path, bbox_inches='tight')
    return

def same_inst_diff_maps(inst, val):
    """
    Diff Map for inst: reach cell to left of heart and maps [85, 327]
    Diff Map for inst: go to the spade . and maps [160, 364]
    """
    same_inst = {}
    for i, s in enumerate(inst):
        for j, d in enumerate(inst):
            if i == j:
                continue
            else:
                if s == d:
                    # print "Found a matching inst: {} {}".format(i, j)
                    if same_inst.has_key(s):
                        if [j, i] not in same_inst[s]:
                            same_inst[s].append([i, j])
                    else:
                        same_inst[s] = []
                        same_inst[s].append([i, j])
    # print same_inst

    for i, maps in same_inst.iteritems():
        for m in maps:
            if np.all(val[m[0]] == val[m[1]]):
                # print "Same Map for inst: {} and maps {}".format(i, m)
                pass
            else:
                print "Diff Map for inst: {} and maps {}".format(i, m)

    return

def same_map_diff_inst(inst, val):
    """
    (85, 202): (u'reach cell to left of heart', u'reach square to the left of heart')
    """
    same_map = {}
    for i, s in enumerate(val):
        for j, d in enumerate(val):
            if i == j:
                continue
            else:
                if np.all(s == d):
                    # print "Found a matching map: {} {}".format(i, j)
                    if (i,j) not in same_map.keys():
                        if (j, i) not in same_map.keys():
                            if inst[i] != inst[j]:
                                same_map[(i,j)] = (inst[i], inst[j])
    # for m, i in same_map.iteritems():
    #     print "{}: {}".format(m, i)
    print len(same_map)

def generate_figures(layouts, objects, save_dir):
    layouts = layouts.squeeze()
    objects = objects.squeeze()
    sprite = environment.SpriteFigure(environment.figure_library.objects,
                                      environment.figure_library.background, dim=100)
    for i, (l, o) in enumerate(zip(layouts, objects)):
        o[o!=0] += 1
        w = (-1*l + 1) + omod
        sprite.makeGrid(w.squeeze(), os.path.join(save_dir, 'world_' + str(i)))
    return

def plot_dqn_values(pkl_path, save_dir, header_name='value_dqn'):
    with open(pkl_path, 'rb') as pid:
        preds = pickle.load(pid).squeeze()
        for i, v in enumerate(preds):
            plt.clf()
            f = plt.figure()
            p = plt.pcolor(v.squeeze(), cmap=cm.jet)
            p.axes.invert_yaxis()
            # f.colorbar(p)
            save_path = os.path.join(save_dir, header_name+'_'+str(i)+'.png')
            plt.savefig(save_path)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', 
                        type=str, 
                        default='', 
                        help='Path to pickle file')
    parser.add_argument('--dqn_pkl', 
                        type=str, 
                        default=None, 
                        help='Path to pickle file')
    parser.add_argument('--save_dir', 
                        type=str, 
                        default=None, 
                        help='Path to pickle file')
    config = parser.parse_args()
    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    visualize_prediction(config.pkl_path, config.save_dir)
    # if config.dqn_pkl:
    #     plot_dqn_values(config.dqn_pkl, config.save_dir)
