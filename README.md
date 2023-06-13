



One of the possible solutions(and also a requirment) is U-net arcitecture. 



We want to know our prediction is accurate or not. For this we can use a DiCE score: 
It estimates a double ratio of intersection of predicted with truth pixel-values and sum of
all predicted and truth pixel-values. 
We will use keras.flatten function to make a 1-dimensional vector from tensor, and after that we 
rewrite formula as double point-wise product between two vectors divided by sum of their square-power(plus some epsiolon)



![U-net arcitecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)



