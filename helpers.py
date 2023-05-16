import numpy as np

def remove_dummy_values(events,dummy_val_list=[-9999,-999,-5,999,9999],is_df=True,
                        return_drop_indeces=False):
  """
  Removes values in df or array from dummy list
  is_df = True for dataframe, false for numpy array (must be 1D)
  """
  if is_df:
    #Remove dummy values from events df
    for val in dummy_val_list:
      events = events[(events != val)]
  else: 
    drop_ind = []
    for val in dummy_val_list:
      drop_condition = np.where(events==val) #Drop all values that are the dummy value
      if len(drop_condition[0]) == 0: continue #skip if we don't find any values to drop
      drop_ind.extend(list(drop_condition[0]))
    events = np.delete(events,drop_ind)
  if return_drop_indeces:
    return events,drop_ind
  else:
    return events