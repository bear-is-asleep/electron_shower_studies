import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
sys.path.append('/sbnd/app/users/brindenc/sbnd_helpers/sbnd_python')
import plotters
import caf

def hist_nreco_vals(pfps,nreco_indices,nreco_labels,key,xlabel,
                    mode='sum',title=None,**kwargs):
  """
  pf particle dataframe
  Supply indices for each reco topology and their labels
  """
  vals = caf.get_nreco_vals(pfps,nreco_indices,key,mode=mode) #get values for each topology
  labels = nreco_labels.copy()
  for i,val in enumerate(vals):
    labels[i] = rf'{labels[i]} ({len(val)})'
  colors = cm.get_cmap('viridis', len(labels))
  colors = [colors(i) for i in range(len(labels))]
  fig,ax = plt.subplots(figsize=(10,6))
  ax.hist(vals,label=labels,histtype='step',color=colors,**kwargs)
  ax.set_xlabel(xlabel)
  ax.legend()
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax
def hist2d_true_reco(reco,true,reco_label,true_label,title=None,plot_line=False,label_boxes=False,
                   **kwargs):
  """
  Plot 2d hist with x axis as reco, y axis as true
  plot_line to plot y=x line
  """
  lower = np.min([0,min(reco),min(true)])
  upper = np.max([max(reco),max(true)])

  fig,ax = plt.subplots(figsize=(6,6),tight_layout=True)

  if plot_line:
    xy = [lower,upper]
    ax.plot(xy,xy,ls='--',color='red')
    ax.set_xlim([lower,upper])
    ax.set_ylim([lower,upper])
    hist,xbins,ybins,im = ax.hist2d(reco,true,range=[xy,xy],**kwargs)
  else:
    hist,xbins,ybins,im = ax.hist2d(reco,true,**kwargs)
  if label_boxes:
    for i in range(len(ybins)-1):
      for j in range(len(xbins)-1):
        ax.text(xbins[j]+0.5,ybins[i]+0.5, f'{hist.T[i,j]:.0f}', 
                color="w", ha="center", va="center", fontweight="bold",fontsize=16)
  ax.set_xlabel(f'{reco_label}')
  ax.set_ylabel(f'{true_label}')
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax

def plot_true_reco_err(reco,true,reco_label,true_label,normalize=True,
                       title=None,**kwargs):
  """
  plot histogram with error between true and reco
  """
  if normalize:
    err = (reco-true)/true
    xlabel = f'({reco_label}-{true_label})/{true_label}'
  else: 
    err = reco-true
    xlabel = f'{reco_label}-{true_label}'
  fig,ax = plt.subplots(figsize=(6,6),tight_layout=True)

  ax.hist(err,**kwargs)
  ax.set_xlabel(xlabel)
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax

def hist_compare_files(dfs,key,file_labels,xlabel,
                    mode='sum',title=None,**kwargs):
  """
  dfs list of dataframes i.e. pfps, mcprims, etc.
  key is key to be plotted
  """
  vals = [caf.get_reco_vals(df,df.index.drop_duplicates(),key,mode=mode)
    for df in dfs] #get values from key and df
  labels = file_labels.copy()
  for i,arr in enumerate(vals):
    labels[i] = rf'{labels[i]} ({len(arr)})'
  colors = cm.get_cmap('viridis', len(dfs))
  colors = [colors(i) for i in range(len(labels))]
  fig,ax = plt.subplots(figsize=(10,6))
  ax.hist(vals,label=labels,histtype='step',color=colors,**kwargs)
  ax.set_xlabel(xlabel)
  ax.legend()
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax