"""
Compare cheated reco from same generation files
"""

import uproot
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/sbnd/app/users/brindenc/sbnd_helpers/sbnd_python')
#My imports
import ttree
import caf_constants
import time
import caf
import plotters
import helpers
import utils

#Constants
NOM_POT = 10e20

mcnu_bad_keys = ['rec.mc.nu.genVersion',	'rec.mc.nu.prim.daughters','rec.mc.nu.wgt.univ']
mcprim_bad_keys = ['rec.mc.nu.prim.daughters']
pfp_bad_keys = ['rec.slc.reco.pfp.shw.truth.matches.G4ID',
                'rec.slc.reco.pfp.shw.truth.matches.energy',
                'rec.slc.reco.pfp.shw.truth.matches.energy_completeness',
                'rec.slc.reco.pfp.shw.truth.matches.energy_purity',
                'rec.slc.reco.pfp.shw.truth.matches.hit_completeness',
                'rec.slc.reco.pfp.shw.truth.matches.hit_purity',
                'rec.slc.reco.pfp.trk.truth.matches.G4ID',
                'rec.slc.reco.pfp.trk.truth.matches.energy',
                'rec.slc.reco.pfp.trk.truth.matches.energy_completeness',
                'rec.slc.reco.pfp.trk.truth.matches.energy_purity',
                'rec.slc.reco.pfp.trk.truth.matches.hit_completeness',
                'rec.slc.reco.pfp.trk.truth.matches.hit_purity'
                ]

#Folder and file names
data_folder = '/sbnd/data/users/brindenc/analyze_sbnd/nue/v09_72_01/'
plots_folder = '/sbnd/app/users/brindenc/analyze_sbnd/nue/reco_studies'
file_labels = ['nocheat','cheat','cheat_vtx']
fnames = ['caf_nue_samegen.root','caf_nue_samegen_cheat.root','caf_nue_samegen_cheat_vtx.root']
sample_pots = [None]*len(fnames)
normalizations = sample_pots.copy()
trees = sample_pots.copy()
mcprims = sample_pots.copy()
pfps = sample_pots.copy()

#Bins
eng_bins = np.arange(0,4,0.1)

#Functions
import matplotlib.cm as cm
def hist_nreco_vals(pfps,nreco_indices,nreco_labels,key,xlabel,
                    mode='sum',title=None,**kwargs):
  """
  pf particle dataframe
  Supply indices for each reco topology and their labels
  """
  vals = caf.get_nreco_vals(pfps,nreco_indices,key) #get values for each topology
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
  plotters.set_style(ax,legend_size=12)
  return fig,ax
def intersection_of_sets(sets):
  """
  Find the indices of the particles in the intersection of three sets.
  """
  intersection = sets[0]
  for i,s in enumerate(sets):
    intersection = intersection.intersection(s)
  #intersection = np.lexsort(list(intersection))
  return list(intersection)
def intersection_of_dfs(dfs):
  """
  Find intersection of list of dataframes. Should be pfps, mcprims, whateva
  """  
  indices_sets = [set(df.index.drop_duplicates()) for df in dfs] #Convert indeces to sets
  common_inds = intersection_of_sets(indices_sets) #Get common inds
  ret_dfs = [df.loc[common_inds] for df in dfs] #Get dataframes with common inds
  return ret_dfs,common_inds
  

#Get tree and POT
for i,fname in enumerate(fnames):
  start = time.perf_counter()
  tree = uproot.open(f'{data_folder}{fname}')
  trees[i] = tree['recTree;1']
  mcprims[i] = ttree.get_mcprim(tree['recTree;1'],
                                bad_keys=mcprim_bad_keys)
  pfps[i] = ttree.get_pfps(tree['recTree;1'],
                            pfp_keys=caf_constants.used_pfp_keys,
                            bad_keys=pfp_bad_keys)
  sample_pots[i] = tree['TotalPOT;1'].to_numpy()[0][0]
  normalizations[i] = NOM_POT/sample_pots[i]
  end = time.perf_counter()
  print(f'Open tree{i} - time = {end-start:.3f} (s)')

#Get common indices - this causes issues for missing tracks so we'll ignore this for now and loop back
#if we really want to do an apples to apples comp.
#mcprims,_ = intersection_of_dfs(mcprims)
#pfps,_ = intersection_of_dfs(pfps)

for i,tree in enumerate(trees):
  start = time.perf_counter()
  #mcnu = ttree.get_mcnu(tree,bad_keys=mcnu_bad_keys)
  mcprim = mcprims[i]
  pfp = pfps[i]
  
  #Drop non intercating primaries
  mcprim,_ = caf.get_interacting_primaries(mcprim)
  
  #Get electrons
  electrons = caf.get_electrons(mcprim)

  #Count tracks and showers
  if i == 1: #cheat pandora
    pfp = caf.count_tracks_showers(pfp,is_cheat=True)
  else:
    #Mask out track scores filled with dummy scores - not reconstructed properly
    _,pfp = caf.mask_dummy_track_score(pfp)
    pfp = caf.count_tracks_showers(pfp)
  
  #Fix bestplane values for showers
  pfp = caf.shw_energy_fix(pfp) 
  #Indeces that match/don't match between truth and reco values
  matched_indices = utils.find_indices_in_common(pfp.index.drop_duplicates(),mcprim.index.drop_duplicates())
  no_reco_indices = utils.find_indices_not_in_common(matched_indices,mcprim.index.drop_duplicates())
  
  #print(fnames[i],[key for key in pfp.keys() if 'nshw' in key])
  #nreco indices
  ntrk0_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.ntrk'] == 0].drop_duplicates()
  ntrk1_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.ntrk'] == 1].drop_duplicates()
  ntrk2_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.ntrk'] >= 2].drop_duplicates()

  nshw0_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.nshw'] == 0].drop_duplicates()
  nshw1_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.nshw'] == 1].drop_duplicates()
  nshw2_indices = pfp.index[pfp.loc[:,'rec.slc.reco.pfp.nshw'] >= 2].drop_duplicates()
  #total_events = utils.flatten_list()


  ntrk0_nshw0_indices = no_reco_indices
  ntrk0_nshw1_indices = utils.find_indices_in_common(ntrk0_indices,nshw1_indices)
  ntrk0_nshw2_indices = utils.find_indices_in_common(ntrk0_indices,nshw2_indices)
  ntrk1_nshw0_indices = utils.find_indices_in_common(ntrk1_indices,nshw0_indices)
  ntrk1_nshw1_indices = utils.find_indices_in_common(ntrk1_indices,nshw1_indices)
  ntrk1_nshw2_indices = utils.find_indices_in_common(ntrk1_indices,nshw2_indices)
  ntrk2_nshw0_indices = utils.find_indices_in_common(ntrk2_indices,nshw0_indices)
  ntrk2_nshw1_indices = utils.find_indices_in_common(ntrk2_indices,nshw1_indices)
  ntrk2_nshw2_indices = utils.find_indices_in_common(ntrk2_indices,nshw2_indices)

  nreco_indices = [
    ntrk0_nshw0_indices,
    ntrk1_nshw0_indices,
    ntrk2_nshw0_indices,
    ntrk0_nshw1_indices,
    ntrk1_nshw1_indices,
    ntrk2_nshw1_indices,
    ntrk0_nshw2_indices,
    ntrk1_nshw2_indices,
    ntrk2_nshw2_indices,
    ]

  nreco_labels = [
    r'$n_{trk}$ = 0 $n_{shw}$ = 0',
    r'$n_{trk}$ = 1 $n_{shw}$ = 0',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ = 0',
    r'$n_{trk}$ = 0 $n_{shw}$ = 1',
    r'$n_{trk}$ = 1 $n_{shw}$ = 1',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ = 1',
    r'$n_{trk}$ = 0 $n_{shw}$ $\geq$ 2',
    r'$n_{trk}$ = 1 $n_{shw}$ $\geq$ 2',
    r'$n_{trk}$ $\geq$ 2 $n_{shw}$ $\geq$ 2',
  ]
  if i == 1:
    print(no_reco_indices)
  
  #Plots
  
  #Reco eng
  fig,ax = hist_nreco_vals(pfp,nreco_indices[1:],nreco_labels[1:],
                'rec.slc.reco.pfp.shw.bestplane_energy',r'$E_{reco} [GeV]$',
                title=f'{file_labels[i]} ({len(electrons.index.drop_duplicates())})',
                bins=eng_bins,
                linewidth=2)
  ax.set_ylim([0,100])
  plotters.set_style(ax)
  plotters.save_plot(f'{file_labels[i]}_reco_eng')

  #True eng
  fig,ax = hist_nreco_vals(electrons,nreco_indices,nreco_labels,
                  'rec.mc.nu.prim.genE',r'$E_{true} [GeV]$',
                  title=f'{file_labels[i]} ({len(electrons.index.drop_duplicates())})',
                  bins=eng_bins,
                  linewidth=2)
  ax.set_ylim([0,100])
  plotters.set_style(ax)
  plotters.save_plot(f'{file_labels[i]}_true_eng')

  end = time.perf_counter()
  print(f'{fnames[i]} - time = {end-start:.3f} (s)')


