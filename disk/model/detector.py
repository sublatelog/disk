import torch
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical, Bernoulli
from torch_dimcheck import dimchecked

from disk import Features, NpArray
from disk.model.nms import nms

@dimchecked
def select_on_last(values: [..., 'T'], indices: [...]) -> [...]:
    '''
    WARNING: this may be reinventing the wheel, but I don't know how to do
    it otherwise with PyTorch.

    This function uses an array of linear indices `indices` between [0, T] to
    index into `values` which has equal shape as `indices` and then one extra
    dimension of size T.
    '''
    
    # torch.gather():指定した軸方向でindicsの値を取って並べる
    return torch.gather(
                        values,
                        -1, # 列方向
                        indices[..., None] # インデックス
                        ).squeeze(-1) # サイズ 1 の input すべての次元が削除されたテンソルを返します。

@dimchecked
def point_distribution(logits: [..., 'T']) -> ([...], [...], [...]):
    
    '''
    Implements the categorical proposal -> Bernoulli acceptance sampling
    scheme. Given a tensor of logits, performs samples on the last dimension,
    
    returning
        a) the proposals
        b) a binary mask indicating which ones were accepted
        c) the logp-probability of (proposal and acceptance decision)
    '''
    
    print("logits")
    print(logits)
    
    # カテゴリ分布は、各カテゴリの確率を個別に指定して、K個の可能なカテゴリの1つをとることができる確率変数の可能な結果を表す離散確率分布です。
    proposal_dist = Categorical(logits=logits)
    
    
    print("proposal_dist")
    print(proposal_dist)
    
    # カテゴリ分布からサンプルを取得
    proposals     = proposal_dist.sample()
    
    
    print("proposals")
    print(proposals)
    
    # カテゴリ分布から取得したサンプルの確立を取得
    proposal_logp = proposal_dist.log_prob(proposals)
    
    
    print("proposal_logp")
    print(proposal_logp)

    # 
    accept_logits = select_on_last(logits, proposals).squeeze(-1)
    
    
    
    print("accept_logits")
    print(accept_logits)

    # ベルヌーイ分布とは、数学において、確率 p で 1 を、確率 q = 1 − p で 0 をとる、離散確率分布である。
    accept_dist    = Bernoulli(logits=accept_logits)
    
    print("accept_dist")
    print(accept_dist)
    
    
    accept_samples = accept_dist.sample()
    print("accept_samples")
    print(accept_samples)
    accept_logp    = accept_dist.log_prob(accept_samples)
    
    print("accept_logp")
    print(accept_logp)
    accept_mask    = accept_samples == 1.
    print("accept_mask")
    print(accept_mask)

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp

class Keypoints:
    '''
    A simple, temporary struct used to store keypoint detections and their
    log-probabilities. After construction, merge_with_descriptors is used to
    select corresponding descriptors from unet output.
    '''

    @dimchecked
    def __init__(self, xys: ['N', 2], logp: ['N']):
        self.xys  = xys
        self.logp = logp

    @dimchecked
    def merge_with_descriptors(self, descriptors: ['C', 'H', 'W']) -> Features:
        '''
        Select descriptors from a dense `descriptors` tensor, at locations
        given by `self.xys`
        '''
        x, y = self.xys.T
        
        """
        descriptors[:, y, x].T
        tensor([[-0.5272,  0.3105,  0.6288,  ..., -0.9846, -0.1811,  0.5275],
                [-0.3851,  0.2575,  0.8689,  ..., -0.2991, -0.0336,  0.2040],
                [-0.2024,  0.4553,  0.5344,  ..., -0.2111,  0.0977,  0.0654],
                ...,
                [ 0.4459,  0.1703,  0.3052,  ..., -0.0518, -0.2994, -0.1908],
                [ 0.4565,  0.0155,  0.3568,  ...,  0.0365,  0.0276,  0.1069],
                [ 0.4613, -0.0645,  0.5657,  ..., -0.1713,  0.2276,  0.1598]],
               device='cuda:0')
        """

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)
        
        """
        desc
        tensor([[-0.0889,  0.0524,  0.1060,  ..., -0.1660, -0.0305,  0.0890],
                [-0.0817,  0.0547,  0.1844,  ..., -0.0635, -0.0071,  0.0433],
                [-0.0428,  0.0963,  0.1130,  ..., -0.0446,  0.0207,  0.0138],
                ...,
                [ 0.1599,  0.0611,  0.1094,  ..., -0.0186, -0.1074, -0.0684],
                [ 0.1588,  0.0054,  0.1241,  ...,  0.0127,  0.0096,  0.0372],
                [ 0.1956, -0.0274,  0.2399,  ..., -0.0726,  0.0965,  0.0678]],
               device='cuda:0')
        """      

        return Features(self.xys.to(torch.float32), desc, self.logp)

class Detector:
    def __init__(self, window=8):
        self.window = window

    @dimchecked
    def _tile(self, heatmap: ['B', 'C', 'H', 'W']) -> ['B', 'C', 'h', 'w', 'T']:
        '''
        Divides the heatmap `heatmap` into tiles of size (v, v) where
        v==self.window. The tiles are flattened, resulting in the last
        dimension of the output T == v * v.
        '''
        v = self.window
        b, c, h, w = heatmap.shape

        assert heatmap.shape[2] % v == 0
        assert heatmap.shape[3] % v == 0
        
        
        print("heatmap.unfold(2, v, v)")
        print(heatmap.unfold(2, v, v))
        
        
        print("heatmap.unfold(2, v, v).unfold(3, v, v)")
        print(heatmap.unfold(2, v, v).unfold(3, v, v))
        
        print("heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v*v)")
        print(heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v*v))

        return heatmap.unfold(2, v, v) \
                      .unfold(3, v, v) \
                      .reshape(b, c, h // v, w // v, v*v)

    @dimchecked
    def sample(self, heatmap: ['B', 1, 'H', 'W']) -> NpArray[Keypoints]:
        '''
            Implements the training-time grid-based sampling protocol
        '''
        v = self.window
        dev = heatmap.device
        B, _, H, W = heatmap.shape

        assert H % v == 0
        assert W % v == 0
        
        
        
        print("heatmap")
        print(heatmap)

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        
        
        print("heatmap_tiled")
        print(heatmap_tiled)
        
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
        )[::-1], dim=0).unsqueeze(0)
        
        
        
        print("cgrid")
        print(cgrid)
        
        cgrid_tiled = self._tile(cgrid)
        
        
        print("cgrid_tiled")
        print(cgrid_tiled)

        # extract xy coordinates from cgrid according to indices sampled
        # before
        xys = select_on_last(
            self._tile(cgrid).repeat(B, 1, 1, 1, 1),
            # unsqueeze and repeat on the (xy) dimension to grab
            # both components from the grid
            proposals.unsqueeze(1).repeat(1, 2, 1, 1)
        ).permute(0, 2, 3, 1) # -> bhw2
         
        keypoints = []
        for i in range(B):
            mask = accept_mask[i]
            keypoints.append(Keypoints(
                xys[i][mask],
                logp[i][mask],
            ))

        return np.array(keypoints, dtype=object)

    @dimchecked
    def nms(
        self,
        heatmap: ['B', 1, 'H', 'W'],
        n=None,
        **kwargs
    ) -> NpArray[Keypoints]:
        '''
            Inference-time nms-based detection protocol
        '''
        heatmap = heatmap.squeeze(1)
        nmsed = nms(heatmap, **kwargs)

        keypoints = []
        for b in range(heatmap.shape[0]):
            yx   = nmsed[b].nonzero(as_tuple=False)
            logp = heatmap[b][nmsed[b]]
            xy   = torch.flip(yx, (1, ))

            if n is not None:
                n_ = min(n+1, logp.numel())
                # torch.kthvalue picks in ascending order and we want to pick in
                # descending order, so we pick n-th smallest among -logp to get
                # -threshold
                minus_threshold, _indices = torch.kthvalue(-logp, n_)
                mask = logp > -minus_threshold

                xy   = xy[mask]
                logp = logp[mask]

            keypoints.append(Keypoints(xy, logp))

        return np.array(keypoints, dtype=object)
