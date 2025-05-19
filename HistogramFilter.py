import numpy as np
import matplotlib.pyplot as plt
import random

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """
    def __init__(self):
        self.eta = 0

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''
        rows,cols = cmap.shape

        prior = belief.flatten()
        obsMatrix =  np.where(cmap == observation, 0.9,0.1).flatten()
        transitionMatrix = np.zeros((rows* cols,rows* cols))
        for i in range(rows):
            for j in range(cols):
                newX = i - action[1]
                newY = j + action[0]
                if 0<= newX < rows and 0 <= newY < cols:
                    transitionMatrix[i*cols + j,newX*cols + newY] += 0.9
                    transitionMatrix[i*cols + j,i*cols + j] += 0.1
                else:
                    transitionMatrix[i*cols + j,i*cols + j] += 1

        alpha = obsMatrix * (transitionMatrix.T @ prior)
        posterior = alpha / np.sum(alpha)
        return posterior.reshape(cmap.shape)
        

if __name__ == "__main__":

    # Load the data
    data = np.load(open('starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    h = HistogramFilter()
    belief = np.full(cmap.shape, 1/(cmap.shape[0] * cmap.shape[1]))

    for t, move in enumerate(actions):
        posterior = h.histogram_filter(cmap, belief, move, observations[t])
        posteriorRot = np.rot90(posterior,-1)
        true_state = belief_states[t]
        predicted_state = np.unravel_index(posteriorRot.argmax(), posteriorRot.shape)

        print("Move: ", move)
        print("Observed: ", observations[t])
        print("True State: ", true_state)
        print("Predicted State: ", predicted_state)

        # Plot the results
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the colormap
        ax[0].imshow(cmap, cmap='gray', interpolation='none')
        ax[0].set_title("Colormap (cmap)")
        #ax[0].scatter(true_state[1], true_state[0], color='red', label='True State')
        #ax[0].legend()

        # Plot the probability distribution map
        ax[1].imshow(posteriorRot, cmap='hot', interpolation='none')
        ax[1].set_title("Probability Distribution (Belief)")
        ax[1].scatter(predicted_state[1], predicted_state[0], color='blue', label='Predicted State')
        ax[1].scatter(true_state[1], true_state[0], color='red', label='True State')
        ax[1].legend()

        plt.show()
        belief = posterior
    