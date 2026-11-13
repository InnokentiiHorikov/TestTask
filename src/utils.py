import pandas as pd
import numpy as np
import torch

def data_preparing(data: pd.DataFrame,
                   to_save_missed: float = 0.02
                   to_downsample: int = 2200,
                   seed: int = 42) -> pd.DataFrame:
  data['has_image_ship'] = data.groupby(['ImageId'])['EncodedPixels'].transform(lambda x: any(pd.notna(x)))
  data['ships_per_image'] = data.groupby(['ImageId'])['EncodedPixels'].transform(lambda x:  sum(pd.notna(x)))
  #Filling na values in the segmentation variable
  data['RLE'] = data['EncodedPixels'].fillna('')
  
  #Downsampling empty images
  missed_ships_ind = data[~data['has_image_ship']].index
  
  rng = np.random.default_rng(seed = seed)
  
  #We can save a negligible part of empty images due to patching images with ship 
  
  
  missed_ships_ind_to_save = rng.choice(missed_ships_ind, 
                                        size=int(to_save_missed*len(missed_ships_ind)), 
                                       replace=False)
  train = data.loc[np.concat([missed_ships_ind_to_save, data[data['has_image_ship']].index])]
  
  #Concatanating all rle's per image
  train = train.groupby('ImageId').agg({"RLE": lambda x: " ".join(x),
                                        "has_image_ship": lambda x: any(x),
                                        "ships_per_image": lambda x: max(x)}).reset_index()
  #Downsample all data
  mask = train['ships_per_image'] <= 3 
  sampled_train_ind = train[mask].groupby('ships_per_image')\
                                 .sample(n=to_downsample, random_state=seed).index
  sampled_train = train.loc[np.concat([sampled_train_ind, train[~mask].index])]
  
  sampled_train.reset_index(drop = True, inplace = True)

  return sampled_train

def RLE_decoding(rle_arr: np.array(str | None), 
                shape: (int, int) = (768, 768)) -> torch.Tensor:
    """
    Input:
    
    Shape: tuple(Height, Width)

    
    Ouput: np.array[1, Height, Width] of RLE masks
    for original(not transformed) 3D Object"
    """
    
    RLE = torch.zeros((1, shape[0], shape[1]))

            
    if (pd.isna(rle_arr) or (rle_arr == '')):
        return RLE
    
    rle = list(map(int, rle_arr.split()))
    start_pixel, len_of_masking = rle[::2], rle[1::2]
    temp = torch.zeros((shape[0]*shape[1]))
            
    masking = torch.hstack(
                [torch.arange(start_pixel[i], start_pixel[i]+len_of_masking[i])
                for i in range(len(start_pixel))])       
    
    temp[masking-1] = 1 
    RLE[0] = temp.reshape((shape[0], shape[1])).T

    return RLE
