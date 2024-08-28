Chess dataset: http://www.computerchess.org.uk/ccrl/4040/
1,700,000 games of top engines.
Filename: CCRL-4040.[1707916].pgn
Using the file "create_games_fen.py", extract 1,700,000 positions from 1,700,000 different games and save them in the file "fens.txt". Time: 5,510 seconds (~1 hour 32 minutes). 

Run 'Tucano-new.exe', type command 'dataset'. It reads the fens, then for each of them a game is played, and search begin. At depth search bigger than 4,a random position is chosen and stored with beta, static evaluation and a label – pruned or not. (10 positions per second. Run in parallel and create fen_labels_combined or run and wait and create fen_labels).
If previous sections ran in parallel, run combine_fen_labels.py to create fen_labels_combined.txt 

Run create_features: read fen_labels (or fen_labels_combined.txt) and create feature vector of length 389 for every fen (2 minutes for 0.640M records ). Attach the static evaluation and the 'beta' value and result is 391 length vector of features and beta value.

Use Neural Network model and train it using "nn.py" for 100 epochs.

Files:

extract_features:
Extract features representation from each chess state.

nn:
Train and test the neural network that recommends about pruning unpromising nodes.
Load balX.pkl and ballabels.pkl as the balanced data that nn will train and validate on.
Save:
	model.ckpt (model.state_dict())
	loss_accuracy.pkl (log of loss and accuracy per epoch)

NeuralNetwork:
Class contains the network architecture.

analize_loss_accuracy:
Plot the training and validation graphs.

evaluate_nn:
Run the trained nn over a sample and return the output - prune or not!

