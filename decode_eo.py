import numpy as np

def decode_ep(img_name, enc_pix):
  if enc_pix = '0':
    return img_name, enc_pix
  
  else:
    enc_pix = enc_pix.split()
    end_point = np.len(enc_pix)
    num_of_pix = enc_pix[::2]
    rep_of_pix = enc_pix[1::2]
    arr_of_pix = [np.arange(
      np.arange(num_of_pix[i], num_of_pix[i] + rep_of_pix[i]))
                  for i in range(lenght)]
                  
      
    return img_name, np.concatenate(arr_of_pix)
  

  
  
  
