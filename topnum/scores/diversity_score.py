from scipy.spatial.distance import pdist
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List,
    Union
)

from .base_custom_score import BaseCustomScore


L2 = 'euclidean'
KL = 'jensenshannon'
KNOWN_METRICS = [L2, KL, 'hellinger', 'cosine']

r'''
Quote from http://arxiv.org/abs/1409.2993

Our treatment makes use of a measure of diversity among
the topics as a criterion to stop generating new topics.
Diversity is widely studied as a criteria in ranking [20], query
suggestion [19] and document summarization [5]. Some existing
work has used diversity as an evaluation criteria for topic
quality [21], but not for selecting the optimum number
of topics (compare across diﬀerent numbers of topics). We
formally deﬁne the diversity among a set of topics as follows:

\text{Diversity}(\Phi) = \frac{2}{T (T-1)}\sum_{i>j} \text{dist}(\phi_{i*}, \phi_{j*})

where K is the number of topics, θi is the word distribution
of the ith topic, and dist(·,·) is a distance function between
the two models. In practice, one can instantiate the distance
function with the L2 distance or the Jensen−Shannon divergence [7]
(the symmetric version of the Kullback−Leibler divergence [7]).
In this work, we simply adopt the L2 distance because it is more
sensitive to the top-ranked words.

Intuitively, if the number of topics is small, the learned topics
tend to be close to the background language model and thus do not
distinguish well between each other. When the number of topics grows,
the granularity of topics becomes finer and the topics become more distinguishable,
thus increasing the diversity. However, when the number of topics
becomes too large, we start to obtain many small topics which may
be too close to each other, which decreases the topic diversity.
Therefore, diversity seems to be a good measure to capture
the right granularity of topics.


[20] Q. Mei, J. Guo, and D. R. Radev. Divrank: the interplay of
prestige and diversity in information networks. In KDD,
pages 1009–1018, 2010.
[19] H. Ma, M. R. Lyu, and I. King. Diversifying query
suggestion results. In AAAI, 2010.
[5] J. Carbonell and J. Goldstein. The use of mmr,
diversity-based reranking for reordering documents and
producing summaries. In SIGIR, pages 335–336, New York,
NY, USA, 1998. ACM.
[21] D. M. Mimno, H. M. Wallach, E. M. Talley, M. Leenders,
and A. McCallum. Optimizing semantic coherence in topic
models. In EMNLP’11, pages 262–272, 2011.
[7] T. M. Cover and J. A. Thomas. Elements of information
theory. Wiley-Interscience, New York, NY, USA, 1991.


'''


class DiversityScore(BaseCustomScore):
    """
    Higher is better

    """
    def __init__(
            self,
            name: str,
            metric: str = L2,
            class_ids: Union[List[str], str] = None,
            closest: bool = False):
        '''
        Parameters
        ----------
        metric
            What metric to use when computing pairwise topic similarity
            Acceptable values are anything inside `KNOWN_METRICS`

            (Actually, supports anything implemented in scipy.spatial.distance,
            but not everything is sanity-checked)
        class_ids
        closest
            if False, the score will calculate average pairwise distance (default)
            if True, will calculate the average distance to the closest topic
        '''

        super().__init__(name)

        metric = metric.lower()

        self._metric = metric
        self._class_ids = class_ids

        self._closest = closest
        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _DiversityScore(self._metric, self._class_ids, self._closest)


class _DiversityScore(BaseTopicNetScore):
    def __init__(self, metric: str, class_ids: Union[List[str], str] = None, closest: bool = False):
        super().__init__()

        metric = metric.lower()

        if metric not in KNOWN_METRICS:
            raise ValueError(f"Unsupported metric {metric}")

        '''
        # test if metric is known to SciPy
        if metric != "hellinger":
            test_matrix = np.reshape(range(4), (2, 2))
            pdist(test_matrix, metric=metric)
        '''

        self._metric = metric
        self._class_ids = class_ids
        self.closest = closest

    def call(self, model: TopicModel):
        phi = model.get_phi(class_ids=self._class_ids).values

        if self._metric == "hellinger":
            matrix = np.sqrt(phi.T)
            condensed_distances = pdist(matrix, metric='euclidean') / np.sqrt(2)
        else:
            condensed_distances = pdist(phi.T, metric=self._metric)

        if self.closest:
            df = pd.DataFrame(
                index=phi.columns, columns=phi.columns,
                data=squareform(condensed_distances)
            )
            # get rid of zeros on the diagonals
            np.fill_diagonal(df.values, float("inf"))

            return df.min(axis=0).mean()

        return condensed_distances.mean()
