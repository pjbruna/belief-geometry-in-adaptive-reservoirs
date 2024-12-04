import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from reservoir import *
from functions import *


### TRAINING DATA ###

input_keys=['man_s','dog_s','walks_v','bites_v','dog_o','man_o','_']
input_dict = {
    'man_s':    [1,0,0,0,0],
    'dog_s':    [0,1,0,0,0],
    'walks_v':  [0,0,1,0,0],
    'bites_v':  [0,0,0,1,0],
    'dog_o':    [0,1,0,0,0],
    'man_o':    [1,0,0,0,0],
    '_':        [0,0,0,0,1]
}

grammarmat=np.zeros((7,7))
grammarmat[0,2] = .75; grammarmat[0,3]=.25;
grammarmat[1,2] = .25; grammarmat[1,3]=.75;
grammarmat[2,4] = .75; grammarmat[2,5]=.25;
grammarmat[3,4] = .25; grammarmat[3,5]=.75;
grammarmat[4,6] = 1; 
grammarmat[5,6] = 1; 
grammarmat[6,0] = .5; grammarmat[6,1]=.5;

train_data = []
train_labels = []
input_key = '_'

for _ in range(4000):
  index1 = input_keys.index(input_key)
  input_key = random.choices(input_keys, weights=grammarmat[index1])[0]
  index2 = input_keys.index(input_key) 

  train_labels.append(input_key)
  train_data.append(input_dict[input_key])


## RUN MODEL ##

model = AdaptiveReservoir(input_nnodes=len(train_data[0]), nnodes=100, p_link=0.1, leak=0.75, lrate_targ=0.01, lrate_wmat=0.1, targ_min=1)
results = model.run(train_data, learn_on=True)

# Cue trained model

cue1 = [input_dict["man_s"], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
cue2 = [input_dict["dog_s"], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

cue1_labels = ["man_s", "NA", "NA", "NA"]
cue2_labels = ["dog_s", "NA", "NA", "NA"]

cue1_results = model.echo(cue1)
cue2_results = model.echo(cue2)

# Compare spike states

res_states = results[0]

cue1_states = cue1_results[0]
cue2_states = cue2_results[0]

cue1_cor_matrix = correlate_states(res_states, cue1_states)
cue2_cor_matrix = correlate_states(res_states, cue2_states)

plot_correlations(cue1_cor_matrix, train_labels, cue1_labels)
plot_correlations(cue2_cor_matrix, train_labels, cue2_labels)


### PCA ###

res_states = np.array(results[0])[:,-100:] # last 100 timesteps
res_states = res_states.T

pca = PCA(n_components=model.nnodes)
principal_components = pca.fit_transform(res_states)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid()
plt.show()

pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]

timesteps = np.arange(res_states.shape[0])
colors = timesteps

plt.figure(figsize=(8, 6))
plt.plot(pc1, pc2, color='gray', alpha=1, linestyle='dotted', linewidth=0.8)
scatter = plt.scatter(pc1, pc2, c=colors, cmap='viridis', alpha=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('PCA Space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Time')
plt.grid()
plt.show()