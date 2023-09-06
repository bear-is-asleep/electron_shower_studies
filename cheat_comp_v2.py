import pandas as pd
import sys
import numpy as np
import sys
import sklearn as sk
import matplotlib.pyplot as plt
sys.path.append('/sbnd/app/users/brindenc/mysbnana_v09_69_00/srcs/sbnana/sbnana/SBNAna/pyana')
from sbnd.constants import *
from sbnd.prism import *
from sbnd.cafclasses.pfp import PFP
from sbnd.cafclasses.nu import NU
from sbnd.cafclasses.mcprim import MCPRIM
from sbnd.cafclasses import object_calc
from sbnd.general import utils
from sbnd.general import plotters
from sbnd.plotlibrary import makeplot
from pyanalib import panda_helpers

DATA_DIR = '/sbnd/data/users/brindenc/analyze_sbnd/nue/v09_72_01'
fnames = ['caf_nue_samegen.df','caf_nue_samegen_cheat.df','caf_nue_samegen_cheat_vtx.df']
names = ['nocheat','cheat','cheat_vtx']

#GEt no cheat data
pfp_nocheat = PFP(pd.read_hdf(f'{DATA_DIR}/{fnames[0]}',key='pfp'))
pfp_nocheat.postprocess()
pfp_nocheat.stats()

nu_nocheat = NU(pd.read_hdf(f'{DATA_DIR}/{fnames[0]}',key='mcnu'))
nu_nocheat.postprocess()

prim_nocheat = MCPRIM(pd.read_hdf(f'{DATA_DIR}/{fnames[0]}',key='mcprim'))
prim_nocheat = prim_nocheat.postprocess()

nocheat_toteng = pfp_nocheat.get_total_reco_energy()
nocheat_mintheta = pfp_nocheat.get_min_theta()
nocheat_electron = pfp_nocheat.get_true_parts_from_pdg(pdg=11,remove_nan=False)

#Cheat data
pfp_cheat = PFP(pd.read_hdf(f'{DATA_DIR}/{fnames[1]}',key='pfp'))
pfp_cheat.postprocess(is_cheat=True)
pfp_cheat.stats()

nu_cheat = NU(pd.read_hdf(f'{DATA_DIR}/{fnames[1]}',key='mcnu'))
nu_cheat.postprocess()

prim_cheat = MCPRIM(pd.read_hdf(f'{DATA_DIR}/{fnames[1]}',key='mcprim'))
prim_cheat = prim_cheat.postprocess()

cheat_toteng = pfp_cheat.get_total_reco_energy()
cheat_mintheta = pfp_cheat.get_min_theta()
cheat_electron = pfp_cheat.get_true_parts_from_pdg(pdg=11,remove_nan=False)

pfp_vtx = PFP(pd.read_hdf(f'{DATA_DIR}/{fnames[2]}',key='pfp'))
pfp_vtx.postprocess()
pfp_vtx.stats()

#Vertex cheat data
nu_vtx = NU(pd.read_hdf(f'{DATA_DIR}/{fnames[2]}',key='mcnu'))
nu_vtx.postprocess()

prim_vtx = MCPRIM(pd.read_hdf(f'{DATA_DIR}/{fnames[2]}',key='mcprim'))
prim_vtx = prim_vtx.postprocess()

vtx_toteng = pfp_vtx.get_total_reco_energy()
vtx_mintheta = pfp_vtx.get_min_theta()
vtx_electron = pfp_vtx.get_true_parts_from_pdg(pdg=11,remove_nan=False)

#Get pfps shared among files
pfps = [pfp_nocheat,pfp_cheat,pfp_vtx]
indices = [list(pfp.index.drop_duplicates()) for pfp in pfps]
shared_indices = utils.common_indices(*indices)
shared_pfps = [PFP(pfp.loc[shared_indices]) for pfp in pfps]

#get shared other objects
shared_totengs = [nocheat_toteng.loc[shared_indices],cheat_toteng.loc[shared_indices],vtx_toteng.loc[shared_indices]]
shared_minthetas = [nocheat_mintheta.loc[shared_indices],cheat_mintheta.loc[shared_indices],vtx_mintheta.loc[shared_indices]]
shared_electrons = [nocheat_electron.loc[shared_indices],cheat_electron.loc[shared_indices],vtx_electron.loc[shared_indices]]

#Even lower level objects
shared_nshw = [pfp[pfp.true_nshw == 1].nshw.groupby(level=[0,1,2]).first() for pfp in shared_pfps]
shared_ntrk = [pfp[pfp.true_nshw == 1].ntrk.groupby(level=[0,1,2]).first() for pfp in shared_pfps]
shared_true_nshw = [pfp[pfp.true_nshw == 1].true_nshw.groupby(level=[0,1,2]).first() for pfp in shared_pfps]
shared_true_ntrk = [pfp[pfp.true_nshw == 1].true_ntrk.groupby(level=[0,1,2]).first() for pfp in shared_pfps]

def split_series_nreco(pfp,mcprim):
  """
  Split a pfp into multiple based on nreco
  """
  no_reco_index = utils.find_indices_not_in_common(
    pfp.index.drop_duplicates(),mcprim.index.drop_duplicates()
  )
  
  #nreco indices
  ntrk0_indices = pfp.index[pfp.ntrk == 0].drop_duplicates()
  ntrk1_indices = pfp.index[pfp.ntrk == 1].drop_duplicates()
  ntrk2_indices = pfp.index[pfp.ntrk >= 2].drop_duplicates()

  nshw0_indices = pfp.index[pfp.nshw == 0].drop_duplicates()
  nshw1_indices = pfp.index[pfp.nshw == 1].drop_duplicates()
  nshw2_indices = pfp.index[pfp.nshw >= 2].drop_duplicates()


  ntrk0_nshw0_indices = no_reco_index
  ntrk0_nshw1_indices = utils.find_indices_in_common(ntrk0_indices,nshw1_indices)
  ntrk0_nshw2_indices = utils.find_indices_in_common(ntrk0_indices,nshw2_indices)
  ntrk1_nshw0_indices = utils.find_indices_in_common(ntrk1_indices,nshw0_indices)
  ntrk1_nshw1_indices = utils.find_indices_in_common(ntrk1_indices,nshw1_indices)
  ntrk1_nshw2_indices = utils.find_indices_in_common(ntrk1_indices,nshw2_indices)
  ntrk2_nshw0_indices = utils.find_indices_in_common(ntrk2_indices,nshw0_indices)
  ntrk2_nshw1_indices = utils.find_indices_in_common(ntrk2_indices,nshw1_indices)
  ntrk2_nshw2_indices = utils.find_indices_in_common(ntrk2_indices,nshw2_indices)
  
  indices = [
      ntrk0_nshw1_indices,
      ntrk0_nshw2_indices,
      ntrk1_nshw0_indices,
      ntrk1_nshw1_indices,
      ntrk1_nshw2_indices,
      ntrk2_nshw0_indices,
      ntrk2_nshw1_indices,
      ntrk2_nshw2_indices,
  ]
  return [pfp.loc[index] for index in indices],indices,no_reco_index
  
#Get pfps as function of reconstructed events
pfp_nocheat_nrecos,nocheat_nreco_indices,nocheat_noreco_indices = split_series_nreco(pfp_nocheat,prim_nocheat[prim_nocheat.in_tpc.values])
pfp_cheat_nrecos,cheat_nreco_indices,cheat_noreco_indices = split_series_nreco(pfp_cheat,prim_cheat[prim_cheat.in_tpc.values])
pfp_vtx_nrecos,vtx_nreco_indices,vtx_noreco_indices = split_series_nreco(pfp_vtx,prim_vtx[prim_vtx.in_tpc.values])

#Store in some lists
pfps_nrecos = [pfp_nocheat_nrecos,pfp_cheat_nrecos,pfp_vtx_nrecos]
nrecos_indices = [nocheat_nreco_indices,cheat_nreco_indices,vtx_nreco_indices]
norecos_indices = [nocheat_noreco_indices,cheat_noreco_indices,vtx_noreco_indices]

#get shared indices
nocheat_nreco_indices_shared = [utils.find_indices_in_common(ind,shared_indices) for ind in nocheat_nreco_indices]
cheat_nreco_indices_shared = [utils.find_indices_in_common(ind,shared_indices) for ind in cheat_nreco_indices]
vtx_nreco_indices_shared = [utils.find_indices_in_common(ind,shared_indices) for ind in vtx_nreco_indices]

nreco_labels = [
    r'$n_{trk}$ = 0 $n_{shw}$ = 0',
    r'$n_{trk}$ = 0 $n_{shw}$ = 1',
    r'$n_{trk}$ = 0 $n_{shw}$ $\geq$ 2',
    r'$n_{trk}$ = 1 $n_{shw}$ = 0',
    r'$n_{trk}$ = 1 $n_{shw}$ = 1',
    r'$n_{trk}$ = 1 $n_{shw}$ $\geq$ 2',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ = 0',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ = 1',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ $\geq$ 2',
  ]

def plot_files_series(series,xlabels,labels,bins,savenames,ylabel=None,title=None,save=False,
                      **pltkwargs):
  for x,xlabel,bi,savename in zip(series,xlabels,bins,savenames):
    #Make plot
    fig,ax = makeplot.plot_hist(x,labels=labels,xlabel=xlabel,bins=bi,**pltkwargs)
    if ylabel is not None:
      ax.set_ylabel(ylabel)
    plotters.set_style(ax,legend_size=12,legend_loc=(1.05,0.1))
    print(savename)
    if save:
      plotters.save_plot(f'{savename}')
      plt.close('all')
  return fig,ax
  
eng_err = [object_calc.get_err(reco,true) for reco,true in zip(shared_totengs,[e.genE for e in shared_electrons])]
theta_diff = [object_calc.get_err(reco,true,normalize=False) for reco,true in zip(shared_minthetas,[e.theta for e in shared_electrons])]

series = [
  # eng_err,
  # theta_diff,
  # eng_err,
  # theta_diff,
]
xlabels = [
  # r'Reco - True / True ($E$)',
  # r'Reco - True ($\theta$)',
  # r'Reco - True / True ($E$)',
  # r'Reco - True ($\theta$)',
]
bins = [
  # 20,
  # 20,
  # np.arange(-0.2,0.1,0.02),
  # np.arange(-0.1,0.5,0.03),
]
savenames = [
  # 'shw_eng_err',
  # 'shw_theta_diff',
  # 'shw_eng_err_zoom',
  # 'shw_theta_diff_zoom',
]

plot_files_series(series,xlabels,names,bins,savenames,lw=2,histtype='step',save=False)

series = [
  #[shared_totengs[0].loc[ind] for ind in nocheat_nreco_indices_shared],
  #[shared_totengs[1].loc[ind] for ind in cheat_nreco_indices_shared],
  #[shared_totengs[2].loc[ind] for ind in vtx_nreco_indices_shared],
  # [shared_minthetas[0].loc[ind] for ind in nocheat_nreco_indices_shared],
  # [shared_minthetas[1].loc[ind] for ind in cheat_nreco_indices_shared],
  # [shared_minthetas[2].loc[ind] for ind in vtx_nreco_indices_shared],
]
xlabels = [
  #r'$E_{reco}$',
  #r'$E_{reco}$',
  #r'$E_{reco}$',
  # r'$\theta_{reco}$',
  # r'$\theta_{reco}$',
  # r'$\theta_{reco}$',
]
bins = [
  #np.arange(0,2,0.2),
  #np.arange(0,2,0.2),
  #np.arange(0,2,0.2),
  # np.arange(0,0.5,0.05),
  # np.arange(0,0.5,0.05),
  # np.arange(0,0.5,0.05),
  
]
savenames = [
  #'shw_eng_nocheat',
  #'shw_eng_cheat',
  #'shw_eng_vtx',
  'shw_theta_nocheat_dens',
  'shw_theta_cheat_dens',
  'shw_theta_vtx_dens',
]

plot_files_series(series,xlabels,nreco_labels[1:],bins,savenames,
                  ylabel='Density',
                  lw=2,histtype='step',
                  density=True,
                  #save=True
                  )

def plot_files_hist2d_series(xseries,xlabels,yseries,ylabels,savenames,bins,titles,save=False,
               **pltkwargs):
  for i,(x,xlabel,y,ylabel,savename,bi,title) in enumerate(zip(xseries,xlabels,yseries,ylabels,savenames,bins,titles)):
    #Make plot
    fig,ax = makeplot.plot_hist2d(x,y,xlabel,ylabel,title=title,bins=bi,
                                  **pltkwargs)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plotters.set_style(ax)
    #Save
    print(savename)
    if save:
      plotters.save_plot(f'{savename}')
      plt.close('all')
  return fig,ax

xs = [
  # shared_totengs[2],
  # shared_totengs[1],
  # shared_totengs[0],
  # shared_totengs[2],
  # shared_totengs[1],
  # shared_totengs[0],
  shared_nshw[0],
  shared_ntrk[0],
  shared_nshw[1],
  shared_ntrk[1],
  shared_nshw[2],
  shared_ntrk[2],
  
]
ys = [
  # shared_electrons[1].genE.dropna(inplace=False),
  # shared_electrons[1].genE.dropna(inplace=False),
  # shared_electrons[0].genE.dropna(inplace=False),
  # shared_electrons[1].genE.dropna(inplace=False),
  # shared_electrons[1].genE.dropna(inplace=False),
  # shared_electrons[0].genE.dropna(inplace=False),
  shared_true_nshw[0],
  shared_true_ntrk[0],
  shared_true_nshw[1],
  shared_true_ntrk[1],
  shared_true_nshw[2],
  shared_true_ntrk[2],
]
xlabels = [
  # r'$E_{reco}$',
  # r'$E_{reco}$',
  # r'$E_{reco}$',
  # r'$E_{reco}$',
  # r'$E_{reco}$',
  # r'$E_{reco}$',
  r'Reco $n_{shw}$',
  r'Reco $n_{trk}$',
  r'Reco $n_{shw}$',
  r'Reco $n_{trk}$',
  r'Reco $n_{shw}$',
  r'Reco $n_{trk}$',
]
ylabels = [
  # r'$E_{true}$',
  # r'$E_{true}$',
  # r'$E_{true}$',
  # r'$E_{true}$',
  # r'$E_{true}$',
  # r'$E_{true}$',
  r'True $n_{shw}$',
  r'True $n_{trk}$',
  r'True $n_{shw}$',
  r'True $n_{trk}$',
  r'True $n_{shw}$',
  r'True $n_{trk}$',
]
savenames = [
  # f'hist2d_eng_{names[2]}',
  # f'hist2d_eng_{names[1]}',
  # f'hist2d_eng_{names[0]}',
  # f'hist2d_eng_{names[2]}_zoom',
  # f'hist2d_eng_{names[1]}_zoom',
  # f'hist2d_eng_{names[0]}_zoom',
  f'confusion_nshw_{names[0]}',
  f'confusion_ntrk_{names[0]}',
  f'confusion_nshw_{names[1]}',
  f'confusion_ntrk_{names[1]}',
  f'confusion_nshw_{names[2]}',
  f'confusion_ntrk_{names[2]}',
]
bins = [
  # 30,
  # 30,
  # 30,
  # np.arange(0,1,0.05),
  # np.arange(0,1,0.05),
  # np.arange(0,1,0.05),
  np.arange(0,7,1),
  np.arange(0,7,1),
  np.arange(0,7,1),
  np.arange(0,7,1),
  np.arange(0,7,1),
  np.arange(0,7,1),
]
titles = [
  # names[2],
  # names[1],
  # names[0],
  # names[2],
  # names[1],
  # names[0],
  names[0],
  names[0],
  names[1],
  names[1],
  names[2],
  names[2],
]

plot_files_hist2d_series(xs,xlabels,ys,ylabels,savenames,bins,titles,label_boxes=True,save=True)



