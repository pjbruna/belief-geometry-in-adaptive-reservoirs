import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from reservoir import *
from functions import *


### TRAINING DATA ###




## RUN MODEL ##

model = AdaptiveReservoir(input_nnodes=len(train_data[0]), nnodes=100, p_link=0.1, leak=0.75, lrate_targ=0.01, lrate_wmat=0.1, targ_min=1)
results = model.run(train_data, learn_on=True)


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