# Character_level_language_model

A character level language model is built using the RNN network with all the functions written from scratch. The data is synthesized by sampling predictions at each time step and passing it to the next RNN-cell unit. The exploding gradient problem has also been fixed using the gradient clipping method.

## Training and Output

Each line of the dataset (one name) is used as one training example.
The model has been trained for 35000 iterations. While the initial few predictions were random and noisy such as ,

```
Iteration: 0, Loss: 23.087336

Nkzxwtdmfqoeyhsqwasjkjvu
Kneb
Kzxwtdmfqoeyhsqwasjkjvu
Neb
Zxwtdmfqoeyhsqwasjkjvu
Eb
Xwtdmfqoeyhsqwasjkjvu


Iteration: 2000, Loss: 27.884160

Liusskeomnolxeros
Hmdaairus
Hytroligoraurus
Lecalosapaus
Xusicikoraurus
Abalpsamantisaurus
Tpraneronxeros

```

But later, the names seemed acceptable. The output is as follows,

```
Iteration: 34000, Loss: 22.447230

Onyxipaledisons
Kiabaeropa
Lussiamang
Pacaeptabalsaurus
Xosalong
Eiacoteg
Troia

```
