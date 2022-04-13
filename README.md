Files:

create_sampling_data.py:
	
	def create_data:
	Use games dataset to extract states and their estimated reward (stockfish evaluation).
	Save: features.npy
		  oracle_evals.npy
		  positions.fens	(using this file later)
	
	def add_vars_to_data:
	Add random beta to each position and label (prune or not) based on beta and oracle value.
	Return X, labels


create_data_alpha_beta_multiple_cores.py:
	
	def create_set:
	Take the positions from positions.fen, shuffle, and for each one run 'play_main' for one move.
	Params should be:
	collect_data = True
	max_beta_data_collect = 5
	search_max_depth = 5
	layer_depth_pruning = 3
	Save:
		data_collected.pkl
		(# each place contains dictionary: {fen: , beta: , pruned: [0,1], value (if pruned): })
	
	def create_X_and_labels:
	Use data_collected.pkl and extract from their X and labels.
	X - features, beta
	label - [0/1]
	Save:
		X.pkl
		labels.pkl
		
	def extract_lens:
	Load X.pkl and save the length of each row (length of x).
	Do that because when using extract_features there are some unique states where length of features is different then regular.
	Save:
		lens.txt
	
	def balance_data:
	Load lens, X, labels.
	Delete rows and labels where length is different than what should be.
	Save same amount of data labeld 0 or 1 (around 200,000 each)
	Save:
		balX.pkl
		ballabels.pkl
	


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


Play:
Simulates chess games where players use some types of pruning methods.


Player:
Class of a player using alpha-beta with neural pruning.


Params:
Contains parameters of the model, training, testing, etc.	

