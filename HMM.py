import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        alphas = []
        alpha = self.Initial_distribution
        for obs in self.Observations:
            alpha = (self.Emission[:,obs].T * (alpha.T @ self.Transition)).flatten()
            alphas.append(alpha)   
        return np.array(alphas)

    def backward(self):
        betas = []
        beta = np.array([1,1])
        for i in range(len(self.Observations)):
            obs = self.Observations[len(self.Observations) - i -1 ]
            beta = self.Transition @ (beta * self.Emission[:, obs]).T
            betas.append(beta)
        return np.array(betas)

    def gamma_comp(self, alpha, beta):
        gammas= []
        for i in range(len(alpha)):
            gamma = (alpha[i] * beta[i])/(np.sum(alpha[i]))
            gamma /= np.sum(gamma)
            gammas.append(gamma)

        return np.array(gammas)
        
    def xi_comp(self, alpha, beta, gamma):
        xis = np.zeros((19,2,2))
        for i in range(self.Observations.shape[0]- 1):
            obs = self.Observations[i+1]

            xi = np.diag(alpha[i]) @  (self.Transition * (self.Emission[:, obs] * beta[i + 1]))

            eta = np.sum(xi)  # Normalize
            xi /= eta 
            xis[i] += xi
        

        return xis

    def update(self, alpha, beta, gamma, xi):

        new_init_state = gamma[0]

        T_prime = np.sum(xi,axis=0) / np.sum(gamma, axis=0).reshape(-1,1)

        la = np.sum(gamma[self.Observations == 0],axis=0).reshape(-1,1)
        ny = np.sum(gamma[self.Observations == 1],axis=0).reshape(-1,1)
        null = np.sum(gamma[self.Observations == 2],axis=0).reshape(-1,1)
        M_prime =  np.hstack([la,ny,null]) / np.sum(gamma,axis=0).reshape(-1,1)

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.sum(alpha[-1])
        self.Emission = M_prime
        self.Transition = T_prime
        self.Initial_distribution = new_init_state
        P_prime = np.sum(self.forward()[-1])
        return P_original/ (P_original +P_prime), P_prime/ (P_original +P_prime)

if __name__ == '__main__':
    obs_matrix = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    trans_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    obs_list = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    init_dist = np.array([[0.5], [0.5]])
    hmm = HMM(obs_list, trans_matrix, obs_matrix, init_dist)
    alpha = hmm.forward()
    beta = hmm.backward()
    gamma = hmm.gamma_comp(alpha,beta)
    xi = hmm.xi_comp(alpha, beta, gamma)
    T_prime, M_prime, new_init_state = hmm.update(alpha,beta,gamma,xi)
    P_original, P_prime  =hmm.trajectory_probability(alpha,beta,T_prime, M_prime, new_init_state)
   