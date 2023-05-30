"""
Multi-Armed bandit problems
---------------------------

R0: List of useful multi-armed bandit resources:
    https://www.kaggle.com/getting-started/131811

R1: Tutorial of Solution Strategies
https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/
https://towardsdatascience.com/solving-multi-armed-bandit-problems-53c73940244a

R3: Building a recommender using machine learning
https://towardsdatascience.com/building-a-recommender-system-using-machine-learning-2eefba9a692e

R4: Q-Learnings
https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/

R5: Deep Q-Lerning sepsis
https://www.nature.com/articles/s41746-023-00755-5#code-availability
https://github.com/CaryLi666/ID3QNE-algorithm

Explore its applicability for recommender systems...

A) Could it be used a strategy to define which antimicrobials suggestions to be shown?

The data considered would be susceptibility test data. Once we know the resistances
of the microorganisms to the available antimicrobials we have to display which
antimicrobials would be recommended. Without any patient related information, this
could be seen as "advertising" narrower spectrum antimicrobials while exploring
how these are being "clicked".


B) Recommender Systems

We have our CBR system which measures similarities between patients. For example,
by using an unsupervised (or self-supervised) approach with a neural network we
can compute the similarity metrics (e.g. demographics, clinical, biomarkers,
radiology, ...). Anyways, based on these we have to retrieve from the database
cases which are similar and show them to the clinicians. By using clinicians feedback
we could see which cases were being clicked and adapt the ones that are being retrieved?

Shouldnt we adapt the model?
=======
Multi Armed Bandit
---------------------------

v1 = [R, R, R, R]
v2 = [R, R, R, R]
>

"""