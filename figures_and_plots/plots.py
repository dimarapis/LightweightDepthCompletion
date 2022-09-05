import numpy as np
import matplotlib

import matplotlib.pyplot as plt


def dual_plot():  
    
    A = np.arange(1, 11)
    B = np.random.randn(10) # 10 rand. values from a std. norm. distr.
    C = B.cumsum()

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,5))
    ## A) via plt.step()



    ax0.step(A, C, label='cumulative sum') # cumulative sum via numpy.cumsum()
    ax0.scatter(A, B, label='actual values')
    ax0.set_ylabel('Y value')
    ax0.legend(loc='upper right')


    ## B) via plt.plot()

    ax1.plot(A, C, label='cumulative sum') # cumulative sum via numpy.cumsum()
    ax1.scatter(A, B, label='actual values')
    ax1.legend(loc='upper right')

    fig.text(0.5, 0.04, 'sample number', ha='center', va='center')
    fig.text(0.5, 0.95, 'Cumulative sum of 10 samples from a random normal distribution', ha='center', va='center')

    return fig




def single_plot():
    d_half_reso = dict()
    d_half_reso['DeepLidar'] = (0.115, 276.0)
    d_half_reso['S2D'] = (0.230, 9.3)
    d_half_reso['NLSPN'] = (0.092, 76.1)
    d_half_reso['Our'] =(0.099, 10.7)
    d_half_reso['Our_small'] = (0.111, 4.0)
    #print(d_half_reso)
    lists = d_half_reso.values()
    #print(lists)
    y_deeplidar,x_deeplidar = d_half_reso['DeepLidar']
    y_s2d,x_s2d  = d_half_reso['S2D']
    y_nlspn,x_nlspn  = d_half_reso['NLSPN']
    y_our, x_our = d_half_reso['Our']
    y_oursmall, x_oursmall = d_half_reso['Our_small']
    
    
    #d_full_reso = dict()
    #d_full_reso['DeepLidar'] = (0.115, 1104.0)
    #d_full_reso['S2D'] = (0.230, 36.3)
    #d_full_reso['NLSPN'] = (0.092, 100.2)
    #d_full_reso['Our'] =(0.099, 42.8)
    #d_full_reso['Our_small'] = (0.111, 16.0)
    #lists = d_full_reso.values()
    #y_ful_deeplidar,x_ful_deeplidar = d_full_reso['DeepLidar']
    #y_ful_s2d,x_ful_s2d  = d_full_reso['S2D']
    #y_ful_nlspn,x_ful_nlspn  = d_full_reso['NLSPN']
    #y_ful_our, x_ful_our = d_full_reso['Our']
    #y_ful_oursmall, x_ful_oursmall = d_full_reso['Our_small']
    
    
    #rmse_list.append(refined_computed_result['rmse']-computed_result['rmse'])
    font_size = 20
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : font_size}

    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=font_size) 
    matplotlib.rc('ytick', labelsize=font_size) 
    #plt.rcParams["font.family"] = "Times New Roman"

    fig,ax = plt.subplots()
    #ax.scatter(x_deeplidar, y_deeplidar, color="gray", marker="s", s=150, label="DeepLidar")
    ax.scatter(x_s2d, y_s2d, color="indianred", marker="o", s=300, label="S2D")
    ax.scatter(x_nlspn, y_nlspn, color="olivedrab", marker="D", s=240, label="NLPSN") 
    ax.scatter(x_our, y_our, color="steelblue", marker="X", s=300, label="Our") 
    ax.scatter(x_oursmall, y_oursmall, color="skyblue", marker="P", s=300, label="Our_small") 
    #ax.scatter(x_ful_deeplidar, y_ful_deeplidar, color="gray", marker="s", s=150, label="DeepLidar")
    #ax.scatter(x_ful_s2d, y_ful_s2d, color="gray", marker="o", s=150, label="S2D")
    #ax.scatter(x_ful_nlspn, y_ful_nlspn, color="gray", marker="D", s=120, label="NLPSN") 
    ##ax.scatter(x_ful_our, y_ful_our, color="gray", marker="X", s=150, label="Our") 
    #ax.scatter(x_ful_oursmall, y_ful_oursmall, color="gray", marker="P", s=150, label="Our_small") 
    
    #ax.scatter(rmse_list_sing, sparse_pts_list_sing, color="red", marker="*")
    ax.set_ylabel("RMSE: Root mean squared error (meters)")
    ax.set_xlabel("MACs: Multiply-accumulate operations (G)")
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 0.37])
    ax.grid()
    ax.legend(loc="upper right")
    #ax.scatter(rmse_list, trendline, marker="*")
    return fig
            
single_plot()
plt.show()
#dual_plot()

    