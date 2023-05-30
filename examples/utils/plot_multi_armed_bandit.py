"""
Multi-Armed bandit problems
---------------------------

.. _R1: https://www.kaggle.com/getting-started/131811
.. _R2: https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/
.. _R3: https://towardsdatascience.com/solving-multi-armed-bandit-problems-53c73940244a
.. _R4: https://towardsdatascience.com/building-a-recommender-system-using-machine-learning-2eefba9a692e
.. _R5: https://www.nature.com/articles/s41746-023-00755-5#code-availability
.. _R6: https://github.com/CaryLi666/ID3QNE-algorithm
.. _R7: https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/

See below for a few resources.

    * `R1`_: List of useful multi-armed bandit resources
    * `R2`_: Tutorial - Reinforcement multi-armed banding (python)
    * `R3`_: Tutorial - Solving multi-armed bandit problems
    * `R4`_: Building a recommender system using ML
    * `R5`_: Nature article
    * `R6`_: GitHub Code
    * `R7`_: Introduction to Deep Q-Learning


Explore the applicability of multi-armed bandit problems for recommender systems.

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

Shouldn't we adapt the model?

"""