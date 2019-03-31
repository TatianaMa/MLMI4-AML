# Introduction

Bayes By Backprop is 
- a variational method inspired by VAEs
- puts distributions over weights instead of units
- training target is ELBO
- uses reparametrization trick to allow for update of variational parameters
- MC to estimate gradient of ELBO
- prediction equivalent to ensemble over infinitely many models
- number of sufficient statistics for the variational posterior is the multiplier for how many more parameters we need to learn
- alternatively can use IVI, in particular KIVI


# RL

- Cast UCI mushrooms as Contextual Bandits
- Can use NNs to approximate the expected reward, a bit like the Q function for an RL task where an episode is a single action always
- The agents all eventually learn to be close to the optimal policy, which requires to classify each mushroom correctly
- All agents first decide to pass on everything, then start eating again, with quite good accuracy. 
(This is because at first they will eat a lot of poisonous mushrooms, and so the expected reward of most mushrooms will be negative for eating)
- BBB does this most effectively
- warmup doesn't really make a difference, in some cases it can be even worse

