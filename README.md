Our task is to train a neural network to detect a presence of ships on given image.
Into input we have train data with encoded pixels. If we have NA-value, it means
no ship on image. Otherwise, ships have been detected. 
To write their place on image in train data have been used a encoded pixels(or rle-encoding).
In this encoding, each odd integer represents the start pixel and each even - difference of start and end pixel in same row




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




