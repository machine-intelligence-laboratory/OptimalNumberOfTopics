import logging
import numpy as np
from scipy.spatial.distance import pdist
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore


L2 = 'euclidean'
KL = "jensenshannon"

_logger = logging.getLogger()

'''

Our treatment makes use of a measure of diversity among
the topics as a criterion to stop generating new topics.
Diversity is widely studied as a criteria in ranking [20], query
suggestion [19] and document summarization [5]. Some existing
work has used diversity as an evaluation criteria for topic
quality [21], but not for selecting the optimum number
of topics (compare across diﬀerent numbers of topics). We
formally deﬁne the diversity among a set of topics as follows:


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


'''

class DiversityScore(BaseCustomScore):
    def __init__(
            self,
            name: str,
            metric: str = L2,
            class_ids: List[str] = None):

        super().__init__(name)

        metric = metric.lower()

        if metric not in [L2, KL]:
            raise ValueError()

        self._metric = metric
        self._class_ids = class_ids

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _DiversityScore(self._metric, self._class_ids)


class _DiversityScore(BaseTopicNetScore):
    def __init__(self, metric: str, class_ids: List[str] = None):
        super().__init__()

        metric = metric.lower()

        if metric not in [L2, KL]:
            raise ValueError()

        self._metric = metric
        self._class_ids = class_ids

    def call(self, model: TopicModel):
        phi = model.get_phi(class_ids=self._class_ids).values

        condensed_distances = pdist(phi.T, metric=self._metric)

        # if you need a DataFrame:
        # from scipy.spatial.distance import squareform
        #df = pd.DataFrame(
        #    index=phi.columns, columns=phi.columns, 
        #    data=squareform(condensed_distances)
        #)

        return condensed_distances.mean()
