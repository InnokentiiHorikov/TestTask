Our task is to train a neural network to detect a presence of ships on given image.
Into input we have train data with encoded pixels. If we have NA-value, it means
no ship on image. Otherwise, ships have been detected. 
To write their place on image in train data have been used a encoded pixels(or rle-encoding).
In this encoding, each odd integer represents the start pixel and each even - difference of start and end pixel in same row

Before decoding rle into masks, we have to read a data, and after that group image by name, because some pictures contains more
than one ship.
```
train_data = pd.read_csv('/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv')
train_data = train_data.groupby("ImageId")["EncodedPixels"].agg(EncodedPixels = lambda x: " ".join(map(str,  x)))
train_data.reset_index("ImageId", inplace = True)
train_data = train_data.to_numpy()
```
We can omit the images without any ship 'cause it wound't have any impact on ship's boundaries. 

##Decoding rle-pixels and making a mask
Let's split given string on separate integers
```
enc_pix = np.asarray(enc_pix.split(), dtype = np.uint32)
```
After that, we create two arrays: first will contains start pixels(i.e. odd elements), second - duration of row
```
num_of_pix = np.array(enc_pix[::2])
rep_of_pix = np.array(enc_pix[1::2])
```
The next step is creating a mask
First of all, our mask is matrix with height and widght are 768. 
```
        size = 768
```
Then we should concatenate all decoded pixels, because into output in previous step we obtained
arrays of separate ranges.
```
        rle = np.concatenate(arr_of_pix)

```
Substacting from all pixels 1 because of python indexing
```
        rle -= 1
```
Than creating the mask(on start one-dimesnional array due to flexibility of assigning), by assigning  decoded-pixels 255(what means white), and other pixels - zero(black) 
```
        plot_mask = np.zeros(square, dtype = np.uint8)
        plot_mask[rle] = 255
```
Reshaping the mask to matrix and normilizing it
```
        square = np.power(size, 2)
        plot_mask = np.reshape(plot_mask, (size, size)).T
        plot_mask = plot_mask / 255
        return plot_mask 
```

And decoding by creating a range for some number to number + corresponding difference. 

One of the possible solutions(and also a requirment) is U-net arcitecture. 


Using Python-generator due to large dataset


We want to know our prediction is accurate or not. For this we can use a DiCE score: 
It estimates a double ratio of intersection of predicted with truth pixel-values and sum of
all predicted and truth pixel-values. 
We will use keras.flatten function to make a 1-dimensional vector from tensor, and after that we 
rewrite formula as double point-wise product between two vectors divided by sum of their square-power(plus some epsiolon)



![U-net arcitecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Resolution of all images is 768x786.  786 can be divided by 32, so we can use a special case of U-net with saving shape
of layer during convolution.
Also in model I used a Dropout layer for better prediction perfomance




