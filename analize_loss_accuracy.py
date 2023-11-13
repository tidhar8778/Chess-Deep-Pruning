import numpy as np
import pickle
from matplotlib import pyplot as plt
from params import  folder_model

file_name = folder_model+'/loss_accuracy.pkl'
# file_name = 'C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\bpruning\\model\\loss_accuracy.pkl'
with open(file_name, 'rb') as f:
    results_dict = pickle.load(f)

# print(results_dict)
train_loss = results_dict['train_loss']
train_accuracy = results_dict['train_accuracy']
val_loss = results_dict['val_loss']
val_accuracy = results_dict['val_accuracy']
length = len(train_loss)
assert length == len(train_accuracy) == len(val_loss) == len(val_accuracy)
range_epochs = np.arange(length)
plt.figure()
# Loss
plt.subplot(221)
plt.plot(range_epochs, train_loss, label='Train')
plt.plot(range_epochs, val_loss, label='Validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.grid(True)
# accuracy
plt.subplot(222)
plt.plot(range_epochs, train_accuracy, label='Train')
plt.plot(range_epochs, val_accuracy, label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.grid(True)

plt.legend()
plt.show()
