import numpy as np

class GHMM:
    """
    base GHMM class
    Accomodates both Generalized HMMs (GHMMs) and standard HMMs
    
    GHMM(dict_labeledTMs, dict_tokens)
    --------------------------------------------
    
    It's assumed that the two dictionaries---for 
    labeled (generalized) transition matrices and the tokens that induce them---
    have the same keys, which range from 0 to 
    alphabet_size-1
    """
    def __init__(self, dict_labeledTMs, dict_tokens):
        self.transition_matrices = dict_labeledTMs
        self.tokens = dict_tokens
        self.int_from_token = {self.tokens[key]: key for key in self.tokens}
        self.latent_dim = dict_labeledTMs[0].shape[0] # dimension of latent space
        self.one = np.ones(self.latent_dim) # Note that this is not generally the right eigenstate of T for a GHMM
        self.alphabet_size = len(dict_tokens.keys())
        T = np.zeros((self.latent_dim, self.latent_dim))
        for x in dict_labeledTMs.keys():
            T+= dict_labeledTMs[x]
        self.T = T
        left_eigstuff = np.linalg.eig(T.transpose())
        right_eigstuff = np.linalg.eig(T)
        for index, eigval in enumerate(left_eigstuff[0]):
            if abs(eigval - 1.) < 0.0000001:
                stationary = left_eigstuff[1][:,index]
        for index, eigval in enumerate(right_eigstuff[0]):
            if abs(eigval - 1.) < 0.0000001:
                tau = right_eigstuff[1][:,index] 
                if abs(tau.sum()) > 0.0000001:
                    tau *= self.latent_dim / tau.sum()    # standardizes normalization; e.g., gives the all-ones vector when possible
        #stationary /= stationary @ tau     # Important for normalizing probabilities!  Independent of whether tau is proportional to all-ones vector
        stationary /= stationary.sum() # supposing HMMs rather than GHMMs!
        self.stationary = stationary.real  # should it be real??
        self.tau = tau.real  # should it be real??
        self.current_distr = self.stationary.copy() 
                
    def sample(self):
        """
        update latent distribution while emmitting a symbol
        """
        x_t = np.random.choice(self.alphabet_size, p= [(self.current_distr @ self.transition_matrices[x]) @ self.tau for x in range(self.alphabet_size)] )  # NOTE: assumes  self.current_distr @ self.tau = 1. 
        new_vec = self.current_distr @ self.transition_matrices[x_t]
        self.current_distr = new_vec / (new_vec @ self.tau)  
        return self.tokens[x_t]
    
    def yield_emissions(self, sequence_len, reset_distr=False):
        if reset_distr: self.current_distr = self.stationary.copy() 
        for _ in range(sequence_len):
            new_token = self.sample()
            yield new_token
        
    def evolve(self, token_key):
        """
        update latent distribution via given token
        """
        new_vec = self.current_distr @ self.transition_matrices[token_key]
        self.current_distr = new_vec / (new_vec @ self.tau)   
        
    def probability(self, word, reset_distr=True):
        """
        returns the probability of the word, given the initial current_distr
        """
        if reset_distr: self.current_distr = self.stationary.copy() 
        prob = 1.
        for token in word: 
            token_int = self.int_from_token[token]
            prob *= (self.current_distr @ self.transition_matrices[token_int]) @ self.tau
            self.evolve(token_int)
        return prob
    
    def supports(self, word, min_prob=1E-9, reset_distr=True):
        if reset_distr: self.current_distr = self.stationary.copy() 
        if self.probability(word) > min_prob: 
            return True
        else: 
            return False



