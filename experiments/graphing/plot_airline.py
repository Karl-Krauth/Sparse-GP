import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def read_file(name):
    l = []
    for line in open(name):
        l.append(float(line))
    return l

def strip_list(l):
    if len(l) > 300:
        l = l[0::7]
    while len(l) < 150:
        l.append(l[-1])
    return l[:150]

def plot_airline(savigp, svi, gp1000, gp2000, linear, name):
    colors = [mpatches.Patch(color='red', label='SAVIGP'),
              mpatches.Patch(color='blue', label='SVI'),
              mpatches.Patch(color='green', label='GP1000'),
              mpatches.Patch(color='yellow', label='GP2000'),
              mpatches.Patch(color='black', label='LINEAR')]
    font = {'size': 19.5}

    matplotlib.rc('font', **font)
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.xlim(xmax=149)
    plt.legend(handles=colors)
    plt.plot(savigp, 'r',
             svi, 'b',
             gp1000, 'g',
             gp2000, 'y',
             linear, 'k')
    plt.show()

def plot_double(obj, val, name):
    font = {'size': 19.5}
    matplotlib.rc('font', **font)

    fig, ax1 = plt.subplots()
    ax1.plot(obj, 'b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NELBO', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(val, 'r')
    ax2.set_ylabel(name, color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax1.set_xlim([0, 149])
    plt.show()

nlpd_savigp = np.empty([5, 150])
nlpd_svi = np.empty([5, 150])
nlpd_gp1000 = np.empty([5, 10])
nlpd_gp2000 = np.empty([5, 10])
# Hard coded results since we have so few numbers here.
nlpd_linear = np.array([5.08685873604, 5.13457483244, 4.97758451918, 5.02466269926, 5.27594624659])
for i in xrange(5):
    nlpd_savigp[i] = np.array(strip_list(read_file("airline" + str(i) + "/nlpd.txt")))
    nlpd_svi[i] = np.array(strip_list(read_file("airline" + str(i) + "/nlpdsvi.txt")))
    nlpd_gp1000[i] = np.array(read_file("airline" + str(i) + "/nlpdsmallgp.txt"))
    nlpd_gp2000[i] = np.array(read_file("airline" + str(i) + "/nlpdlargegp.txt"))
    plot_airline(nlpd_savigp[i],
                 nlpd_svi[i],
                 nlpd_gp1000[i].mean().repeat(150),
                 nlpd_gp2000[i].mean().repeat(150),
                 nlpd_linear[i].repeat(150),
                 'NLPD')
plot_airline(nlpd_savigp.mean(axis=0),
             nlpd_svi.mean(axis=0),
             nlpd_gp1000.mean().repeat(150),
             nlpd_gp2000.mean().repeat(150),
             nlpd_linear.mean().repeat(150),
             'NLPD')


rmse_savigp = np.empty([5, 150])
rmse_svi = np.empty([5, 150])
rmse_gp1000 = np.empty([5, 10])
rmse_gp2000 = np.empty([5, 10])
rmse_linear = np.array([38.5742766187, 40.8273352476, 34.578759036, 36.7876864506, 45.3542697617])
for i in xrange(5):
    rmse_savigp[i] = np.array(strip_list(read_file("airline" + str(i) + "/rmse.txt")))
    rmse_svi[i] = np.array(strip_list(read_file("airline" + str(i) + "/rmsesvi.txt")))
    rmse_gp1000[i] = np.array(read_file("airline" + str(i) + "/rmsesmallgp.txt"))
    rmse_gp2000[i] = np.array(read_file("airline" + str(i) + "/rmselargegp.txt"))
    plot_airline(rmse_savigp[i],
                 rmse_svi[i],
                 rmse_gp1000[i].mean().repeat(150),
                 rmse_gp2000[i].mean().repeat(150),
                 rmse_linear[i].repeat(150),
                 'RMSE')
plot_airline(rmse_savigp.mean(axis=0),
             rmse_svi.mean(axis=0),
             rmse_gp1000.mean().repeat(150),
             rmse_gp2000.mean().repeat(150),
             rmse_linear.mean().repeat(150),
             'RMSE')

obj1 = np.array(strip_list(read_file("airline1/obj.txt")))
plot_double(obj1, rmse_savigp[1], "RMSE")
plot_double(obj1, nlpd_savigp[1], "NLPD")
