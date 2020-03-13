import numpy as np
from utility_functions import *

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

chars = sorted(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
 
    for gradient in [dWaa, dWax, dWya, db, dby]:
        gradient = np.clip(gradient,-maxValue,maxValue,out=gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    """
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    
    indices = []
    idx = -1 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        
        np.random.seed(counter+seed) 
        idx = np.random.choice(range(vocab_size),p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        a_prev = a
        seed += 1
        counter +=1
        
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    """
    loss, cache = rnn_forward(X,Y,a_prev,parameters)

    gradients, a = rnn_backward(X,Y,parameters,cache)
    gradients = clip(gradients,5)

    parameters = update_parameters(parameters,gradients,learning_rate)
        
    return loss, gradients, a[len(X)-1]

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names. 
    
    """
    n_x, n_y = vocab_size, vocab_size
    
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    loss = get_initial_loss(vocab_size, dino_names)
    
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):

        idx = j%len(examples)
        
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None]+single_example_ix
        
        ix_newline = char_to_ix['\n']
        Y = X[1:]+[ix_newline]

        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,learning_rate=0.01)
        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            seed = 0
            for name in range(dino_names):

                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1 
            print('\n')
        
    return parameters

parameters = model(data, ix_to_char, char_to_ix)