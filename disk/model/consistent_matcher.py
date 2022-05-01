import torch
from torch import nn
from torch.distributions import Categorical
from torch_dimcheck import dimchecked

from disk import Features, NpArray, MatchDistribution
from disk.geom import distance_matrix

# train > loss_fn.accumulate_grad > ConsistentMatcher_match_pair > ConsistentMatchDistribution
class ConsistentMatchDistribution(MatchDistribution):
    def __init__(
        self,
        features_1: Features,
        features_2: Features,
        inverse_T: float,
    ):
        self._features_1 = features_1
        self._features_2 = features_2
        self.inverse_T = inverse_T
        
        print("self.features_1().desc")
        print(self.features_1().desc.size())
        print(self.features_1().desc)
        

        # distance_matrix(): 1.414213 * (1. - fs1 @ fs2.T).clamp(min=1e-6).sqrt()
        distances = distance_matrix(
            self.features_1().desc,
            self.features_2().desc,
        )
        
        """
        distances
        torch.Size([4660, 4672])
        tensor([[0.8001, 0.9821, 1.0317,  ..., 1.5377, 1.5198, 1.4498],
                [1.2176, 1.1543, 1.1861,  ..., 1.4920, 1.4388, 1.3953],
                [0.9011, 0.7790, 0.8318,  ..., 1.4034, 1.3502, 1.3481],
                ...,
                [1.3575, 1.2276, 1.1596,  ..., 0.7947, 0.5773, 1.0236],
                [1.3981, 1.2601, 1.2054,  ..., 0.8923, 0.7021, 1.0499],
                [1.3091, 1.1891, 1.1679,  ..., 0.8867, 0.6355, 1.0838]],
               device='cuda:0', grad_fn=<MulBackward0>)
        """
       
        
        """
        inverse_T
        tensor(15., device='cuda:0')
        """
        
        affinity = -inverse_T * distances
        """
        affinity
        torch.Size([4660, 4672])
        tensor([[-12.0015, -14.7314, -15.4759,  ..., -23.0651, -22.7974, -21.7463],
                [-18.2635, -17.3140, -17.7920,  ..., -22.3796, -21.5814, -20.9297],
                [-13.5168, -11.6845, -12.4776,  ..., -21.0507, -20.2523, -20.2213],
                ...,
                [-20.3618, -18.4133, -17.3935,  ..., -11.9206,  -8.6591, -15.3540],
                [-20.9710, -18.9022, -18.0805,  ..., -13.3847, -10.5312, -15.7485],
                [-19.6366, -17.8368, -17.5178,  ..., -13.3001,  -9.5326, -16.2570]],
               device='cuda:0', grad_fn=<MulBackward0>)
        """

        self._cat_I = Categorical(logits=affinity)
        self._cat_T = Categorical(logits=affinity.T)

        self._dense_logp = None
        self._dense_p    = None

    # 
    @dimchecked
    def dense_p(self) -> ['N', 'M']:
        if self._dense_p is None:
            self._dense_p = self._cat_I.probs * self._cat_T.probs.T

        return self._dense_p

    @dimchecked
    def dense_logp(self) -> ['N', 'M']:
        if self._dense_logp is None:
            self._dense_logp = self._cat_I.logits + self._cat_T.logits.T

        return self._dense_logp

    @dimchecked
    def _select_cycle_consistent(self, left: ['N'], right: ['M']) -> [2, 'K']:
        indexes = torch.arange(left.shape[0], device=left.device)
        cycle_consistent = right[left] == indexes

        paired_left = left[cycle_consistent]

        return torch.stack([
            right[paired_left],
            paired_left,
        ], dim=0)

    @dimchecked
    def sample(self) -> [2, 'K']:
        samples_I = self._cat_I.sample()
        samples_T = self._cat_T.sample()

        return self._select_cycle_consistent(samples_I, samples_T)

    @dimchecked
    def mle(self) -> [2, 'K']:
        maxes_I = self._cat_I.logits.argmax(dim=1)
        maxes_T = self._cat_T.logits.argmax(dim=1)

        # FIXME UPSTREAM: this detachment is necessary until the bug is fixed
        maxes_I = maxes_I.detach()
        maxes_T = maxes_T.detach()

        return self._select_cycle_consistent(maxes_I, maxes_T)

    def features_1(self) -> Features:
        return self._features_1

    def features_2(self) -> Features:
        return self._features_2

class ConsistentMatcher(torch.nn.Module):
    def __init__(self, inverse_T=1.):
        super(ConsistentMatcher, self).__init__()
        self.inverse_T = nn.Parameter(torch.tensor(inverse_T, dtype=torch.float32))

    def extra_repr(self):
        return f'inverse_T={self.inverse_T.item()}'

    def match_pair(self, features_1: Features, features_2: Features):
        return ConsistentMatchDistribution(features_1, features_2, self.inverse_T)
