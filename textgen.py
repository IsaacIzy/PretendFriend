import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
def prepare_data(filename, verbose=False, seq_len=100):
    '''
    prepare_data helps prepare a text file for use within a neural network.
    This function prints some information about the data, and creates dictionaries
    to help convert the characters in the text file to numerical values so the 
    NN can use them effectively.
    
    Args: 
    filename: location of ASCII file to read
    verbose: print extra information about the data
    seq_len: length of the sequences to prepare
    
    Return:
    char-to-int, int-to-char: dictionaries for converting chars to ints and vice versa
    chars: list of unique characters in the dataset
    X: matrix with all the input sequences
    Y: matrix with all the outputs corresponding to input sequence
    '''
    data = open(filename, 'r').read()
    chars = list(set(data))
    char_to_int = {char:ix for ix, char in enumerate(chars)}
    int_to_char = {ix:char for ix, char in enumerate(chars)}
    if verbose==True:
        print("length of data:", len(data), "number unique chars:", len(chars))
        print("unique chars:")
        print(chars)
    # Prepare the sequences for input into our RNN
    # Note that X will not contain the first <seq_len> characters
    dataX = [] # Input sequences (len(data)/seq_len X seq_len matrix)
    dataY = [] # Ouput sequence (len(data) X 1 matrix)
    for i in range(0, len(data) - seq_len, 1):
        seq_in = data[i:i + seq_len]
        seq_out = data[i + seq_len]
        # make sure to convert the characters to ints
        dataX.append([char_to_int[char] for char in seq_in]) 
        dataY.append(char_to_int[seq_out])
    X = np.array(dataX)
    Y = np.array(dataY)
    return X, Y, char_to_int, int_to_char, len(chars)

def generate(model, seed, vocab_len, int_to_char, length=100):
    '''
    This function generates characters based on the model and seed that it is given.
    
    args
    model: compiled keras model
    length: number of characters to generate, default 100
    
    returns a string of generated characters
    '''
    result = ""
    for i in range(length):
        # reshape the seed into the input the dimensions our model expects
        X = np.reshape(seed, (1, seed.shape[0], 1))
        # normalize
        X = X/float(vocab_len)
        # Make a prediction
        predict = model.predict(X)
        # Pick the value with the highest probability and convert it to a character, and add it to the result
        index = np.argmax(predict)
        char = int_to_char[index]
        result = result + char
        # Append the prediction to the seed, and then remove the first element of the seed.
        # This shifts the "window" over so our model can make a new prediction, also using the
        # new character we just generated. This continues for as long as we have more characters to generate
        seed = np.append(seed, index)
        seed = np.delete(seed, 0)
    return result

if __name__=="__main__":
	print(sys.argv)
	if len(sys.argv) < 4:
		print("Usage: textgen.py <checkpoint file> <data file> <output length>")
		exit(0)
	# Load the data file for statistics and seeding
	dataX,dataY,char_to_int,int_to_char,vocab_len=prepare_data(sys.argv[2])
	# Load the model specified
	model = Sequential()
	model.add(LSTM(250, input_shape = (dataX.shape[1],1), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(250))
	model.add(Dropout(0.2))
	model.add(Dense(vocab_len, activation='softmax'))
	model.load_weights(sys.argv[1])
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# pick a random sequence from the data
	rand = np.random.randint(0, dataX.shape[0]-1)
	seed = dataX[rand]
	print(for value in seed int_to_char[value])
	result = generate(model, seed, vocab_len, int_to_char, 
		length=int(sys.argv[3])) 
	print(result) 
