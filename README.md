# PretendFriend
## Character level text generation using a neural network
This began as a project for an intro to machine learning course I took at CSU. I wanted to explore human like text generation when I saw the subreddit [/r/subredditsimulator](reddit.com/r/subredditsimulator).

I have decided to rexplore this idea for an independent study, as my initial exploration was not as good as I had hoped it would be, mostly do to how long the models were taking to train on my GTX960. I now have new hardware that should make training much faster. 

I have opted to use Keras for my neural network because of how easy it is to try out lots of different model configurations. It is very easy to tweak the number of layers, how deep the layers are, and even add dropout between any layer to reduce overfitting problems. 

My first Iteration of this project can be found in the TextGeneration.ipynb notebook. It describes how I prepare the data, and all the different network configurations I tried. I suggest either using nbviewer or downloading the botebook to your local machine so you can hide some of the long outputs. 

The source code for the first iteration is in textgen.py

I am starting a new notebook and new source files that will be easier for others to use in their own projects. I have chose to leave the original implementation untouched so that others can see the path I took figuring out how to get this up and running. 
