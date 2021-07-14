#-------------------------------------------------------------------------------
# Name:        generateHeatmap
# Purpose:
#
# Author:      yzlco
#
# Created:     01/12/2019
# Copyright:   (c) yzlco 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

#http://seaborn.pydata.org/generated/seaborn.clustermap.html
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import rcParams

class generateHeatmap():
    def __new__(self,mtx,deg_names,pag_ids,**kwargs):
        #matplotlib.use('Agg')
        # Plot it out
        #fig = plt.figure()
        
        length = len(pag_ids)*0.43+1
        #print(length)
        fig, ax = plt.subplots(figsize=(5, length))
        
        # parameters in the heatmap setting 
        width_ratio = 0.8
        if 'width_ratio' in kwargs.keys():
            width_ratio = kwargs['width_ratio']
        
        annotationSize = 5
        if 'annotationSize' in kwargs.keys():
            annotationSize = kwargs['annotationSize']
            
        outputdir = kwargs['outputdir'] if 'outputdir' in kwargs.keys() else ""
        
        col_linkage = hc.linkage(sp.distance.pdist(mtx.T), method='ward')
        row_linkage = hc.linkage(sp.distance.pdist(mtx), method='ward')
        
        # load the color scale using the cm
        #top = cm.get_cmap('Blues_r', 56)
        bottom = cm.get_cmap('Reds', 56)
        newcolors = np.vstack(
                                (
                                    #top(np.linspace(0, 1, 56)),
                                    ([[0,0,0,0.1]]),
                                    bottom(np.linspace(0, 1, 56))
                                )
                             )
        newcmp = ListedColormap(newcolors, name='RedBlue')
        # set the balance point of the expression to 0
        f_max = np.max(mtx)
        f_min = np.min(mtx)
        
        if(abs(f_max)>abs(f_min)):
            Bound=abs(f_max)
        else:
            Bound=abs(f_min)
           
        # figure size in inches
        #rcParams['figure.figsize'] = 3,8.27
        expMtxsDF = pd.DataFrame(mtx)
        expMtxsDF.columns = deg_names
        expMtxsDF.index = pag_ids
        #sns.set(font_scale=1,rc={'figure.figsize':(3,8.27)})
        
        #print(expMtxsDF)
        
        print('rowCluster' in kwargs.keys() & 'colCluster' in kwargs.keys())
        if('rowCluster' in kwargs.keys() & 'colCluster' in kwargs.keys()): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,row_linkage=row_linkage,
                          annot=True,annot_kws={"size": 10},cbar_kws={'label':10 })        
        elif('rowCluster' in kwargs.keys()): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,row_linkage=row_linkage,
                          annot=True,annot_kws={"size": 10},cbar_kws={'label':10 })
        elif('colCluster' in kwargs.keys()): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,
                          annot=True,annot_kws={"size": 10},cbar_kws=dict(use_gridspec=False,pad=0.01,shrink=0.15,label=10,orientation='horizontal'))  
        else:
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,row_cluster=False,
                          annot=True,annot_kws={"size": 10},cbar_kws={'label':10 })  
            #g.ax_row_dendrogram.set_xlim([0,0])
            
        #plt.subplots_adjust(top=0.9) # make room to fit the colorbar into the figure
        rotation = 90 
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 10)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 10)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=10)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=10)
        hm = g.ax_heatmap.get_position()
        # to change the legends location
        
        
        g.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*width_ratio, hm.height])
        col = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*width_ratio, col.height*0.5])
        #for i, ax in enumerate(g.fig.axes):   ## getting all axes of the fig object
        #    ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
        
        bottom, top = g.ax_heatmap.get_ylim()
        
        #g.ax_heatmap.set_ylim(bottom + 0.5, top - 0.5)
        #g.legend(bbox_to_anchor= (1.2,1))
        #leg = g._legend
        #leg.set_bbox_to_anchor([0.5, 0.5])  # coordinates of lower left of bounding box
        #leg._loc = 2  # if required you can set the loc
        plt.rcParams["axes.grid"] = False
        
        plt.show()
        
        
        ## create heatmap using imshow
        ##ax = fig.add_subplot(111)
        #im = ax.imshow(mtx, cmap=plt.cm.Blues,aspect="auto")
        ## We want to show all ticks...
        #ax.set_xticks(np.arange(len(deg_names)))
        #ax.set_yticks(np.arange(len(pag_ids)))
        ## ... and label them with the respective list entries
        #ax.set_xticklabels(deg_names)
        #ax.set_yticklabels(pag_ids)
        ##ax.xaxis.tick_top()
        ## create color bar
        #cbar = ax.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel("-log(P)", rotation=-90, va="bottom")
        ## Rotate the tick labels and set their alignment.
        ## plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  rotation_mode="anchor")
        #plt.xticks(rotation=90,fontsize=18)
        #plt.yticks(fontsize=18)
        #for t in ax.xaxis.get_major_ticks():
        #    t.tick1On = False
        #    t.tick2On = False
        #for t in ax.yaxis.get_major_ticks():
        #    t.tick1On = False
        #    t.tick2On = False        
        ##plt.show()
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        #plt.xlabel('xlabel', fontsize=18)
        #plt.ylabel('ylabel', fontsize=16)
        #plt.savefig("./heatmap.png", dpi=100,transparent = True, bbox_inches = 'tight', pad_inches = 0)        
        # Loop over data dimensions and create text annotations.
##        for i in range(len(pag_ids)):
##            for j in range(len(deg_names)):
##                text = ax.text(j, i, mtx[i, j],ha="center", va="center", color="w")
        #ax.set_title("sample-PAG associations")

        return(plt)
if __name__ == '__main__':
    generateHeatmap()
