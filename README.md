# Bayesian Filtering and Hidden Markov Model Scripts

This repository contains two Python scripts implementing fundamental concepts in probabilistic reasoning for robotics and time series modeling as part of my coursework for ESE 6500: Learning in Robotics:

---

##  `HistogramFilter.py`

This script implements a **Histogram Filter** — a discrete Bayes Filter used for robot localization in a grid-based environment.

###  How it works:
- A binary grid map (`cmap`) represents the environment.
- The robot starts with a uniform belief distribution.
- It receives a sequence of **actions** (movements) and **observations** (color sensor readings).
- The belief distribution is updated over time using:
  - A **motion model** (includes some uncertainty in movement).
  - An **observation model** (includes sensor noise).
- After each update, the script:
  - Displays the predicted vs. true robot location.
  - Visualizes the belief distribution as a heatmap.
---

##  `HMM.py`

This script implements a basic **Hidden Markov Model (HMM)** framework, including training updates and probability evaluations.

###  Features:
- **Forward algorithm**: Computes the probability of an observation sequence.
- **Backward algorithm**: Supports smoothing and training steps.
- **Gamma and Xi computation**: Estimates the probability of being in a state or transitioning between states.
- **Parameter update**: Re-estimates transition and emission matrices based on observed data.
- **Trajectory probability comparison**: Compares the likelihood of the sequence before and after updating model parameters.

###  Example:
A test sequence of observations is provided, and the script:
1. Initializes the HMM with uniform probabilities.
2. Runs forward/backward passes.
3. Updates the model.
4. Compares original vs. updated sequence likelihoods.


---

##  Files

- `HistogramFilter.py` – Grid-based Bayes Filter for localization.
- `HMM.py` – Hidden Markov Model implementation and training demo.
