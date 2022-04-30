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
    
    """
    logits
    torch.Size([3, 96, 96, 64])
    tensor([[[[-3.1139e-01, -6.0347e-01,  8.3655e-02,  ...,  4.8296e-01, 6.0503e-01,  6.6835e-01],
              [ 5.9942e-02,  1.5390e-01,  1.3652e-01,  ...,  6.1927e-01, 6.7140e-01,  7.0280e-01],
              [ 7.9184e-02,  1.0621e-01,  1.6106e-01,  ...,  8.9157e-01, 9.1781e-01,  9.2466e-01],
              ...,
              [ 2.1664e-01,  1.9108e-01,  1.7046e-01,  ...,  2.6707e-01, 3.0826e-01,  3.3816e-01],
              [ 1.7556e-01,  1.5038e-01,  1.3575e-01,  ...,  3.7192e-01, 2.4402e-01,  1.6215e-02],
              [-1.0751e-01, -9.8728e-02, -8.1865e-02,  ...,  1.5102e-01, -1.8443e-01,  1.0855e-01]],
    """
    
    
    # カテゴリ分布は、各カテゴリの確率を個別に指定して、K個の可能なカテゴリの1つをとることができる確率変数の可能な結果を表す離散確率分布です。
    proposal_dist = Categorical(logits=logits)
    
    """
    proposal_dist
    Categorical(logits: torch.Size([3, 96, 96, 64]))
    """
    
    # カテゴリ分布からサンプルを取得
    proposals     = proposal_dist.sample()
    """
    proposals
    torch.Size([3, 96, 96])
    tensor([[[25, 25,  4,  ..., 49,  3, 18],
             [62, 51,  7,  ..., 47, 36, 46],
             [34, 11, 57,  ..., 37, 36, 39],
             ...,
             [53, 19, 54,  ..., 27, 59, 59],
             [47, 23, 32,  ..., 20,  6, 19],
             [18, 37, 40,  ..., 55, 36, 22]],

            [[56, 24, 56,  ...,  6, 61, 11],
             [30, 26, 30,  ..., 56, 17,  2],
             [ 3, 28, 27,  ..., 42, 63, 45],
             ...,
             [14, 42, 37,  ..., 39, 29, 24],
             [38, 16, 61,  ..., 38, 26,  8],
             [27,  5, 47,  ..., 24, 34, 42]],

            [[20, 47, 25,  ..., 25, 48,  4],
             [46, 19, 52,  ..., 24, 22, 11],
             [24, 34,  8,  ..., 40, 12, 36],
             ...,
             [38, 58,  8,  ...,  7, 43, 22],
             [40, 29, 55,  ...,  0, 59, 26],
             [12, 58, 25,  ..., 41,  2,  8]]], device='cuda:0')
    """

    
    # カテゴリ分布から取得したサンプルの確立を取得
    proposal_logp = proposal_dist.log_prob(proposals)
    
    """
    proposal_logp
    torch.Size([3, 96, 96])
    tensor([[[-4.6688, -4.2400, -4.8154,  ..., -4.3598, -4.5061, -3.6803],
             [-4.2810, -4.0045, -4.0672,  ..., -4.0765, -4.0340, -4.1497],
             [-3.8270, -4.1134, -4.1749,  ..., -4.1560, -4.1804, -4.0086],
             ...,
             [-4.0344, -4.2146, -4.1140,  ..., -4.1522, -4.1564, -4.1764],
             [-4.2047, -4.2114, -4.2089,  ..., -4.0213, -4.0845, -4.1149],
             [-4.1996, -4.1805, -4.0962,  ..., -4.0423, -4.1267, -4.1911]],

            [[-4.1045, -3.8476, -4.4524,  ..., -4.2793, -4.1603, -4.2925],
             [-4.1262, -3.9271, -4.1066,  ..., -4.1268, -4.2638, -4.5165],
             [-3.9153, -4.2256, -4.1699,  ..., -4.1498, -4.2517, -3.9270],
             ...,
             [-3.8794, -4.0661, -4.2133,  ..., -4.1579, -4.2239, -4.1598],
             [-3.9908, -3.6833, -4.2530,  ..., -4.2157, -4.1675, -4.1798],
             [-4.0905, -4.4638, -4.0754,  ..., -4.3716, -4.1032, -4.0279]],

            [[-3.3008, -4.7034, -4.3057,  ..., -4.4083, -4.1949, -4.7893],
             [-3.9952, -4.0816, -4.2176,  ..., -4.1332, -4.6029, -3.5130],
             [-4.7883, -4.1256, -4.1884,  ..., -3.9397, -3.8929, -4.0244],
             ...,
             [-4.0980, -4.1818, -4.1366,  ..., -4.1583, -4.1214, -4.0483],
             [-4.3576, -4.2519, -4.2132,  ..., -4.0901, -4.2221, -4.2203],
             [-4.1318, -3.8697, -4.2375,  ..., -4.1077, -4.3045, -4.4504]]],
           device='cuda:0', grad_fn=<SqueezeBackward1>)
    """


    # 
    accept_logits = select_on_last(logits, proposals).squeeze(-1)
    
    """
    accept_logits
    torch.Size([3, 96, 96])
    tensor([[[ 2.1563e-01,  7.7110e-01,  2.6007e-01,  ...,  2.7901e-01, 1.5811e-01,  7.0802e-01],
             [ 3.0520e-01,  5.9063e-01,  5.6615e-01,  ...,  2.9011e-01, 3.4100e-01, -7.7148e-02],
             [ 8.3748e-01,  5.7021e-01,  5.7831e-01,  ...,  1.3027e-01, 9.3538e-02,  6.3673e-02],
             ...,
             [ 1.7103e-01, -4.5518e-01, -4.9895e-01,  ..., -3.2386e-01, -4.6719e-01, -6.1792e-01],
             [-1.3217e-01, -5.0198e-01, -5.8203e-01,  ..., -4.2708e-01, -5.2666e-01, -5.8571e-01],
             [-3.0306e-01, -6.0801e-01, -5.4103e-01,  ..., -6.3684e-01, -5.6356e-01, -6.0317e-01]],
    """


    # ベルヌーイ分布とは、数学において、確率 p で 1 を、確率 q = 1 − p で 0 をとる、離散確率分布である。
    accept_dist    = Bernoulli(logits=accept_logits)
    """
    accept_dist
    Bernoulli(logits: torch.Size([3, 96, 96]))
    """
    
    accept_samples = accept_dist.sample()
    """
    accept_samples
    torch.Size([3, 96, 96])
    tensor([[[0., 0., 1.,  ..., 1., 1., 1.],
             [1., 1., 0.,  ..., 0., 0., 1.],
             [1., 0., 1.,  ..., 1., 0., 1.],
             ...,
             [0., 0., 0.,  ..., 0., 1., 1.],
             [1., 0., 1.,  ..., 0., 0., 0.],
             [0., 0., 1.,  ..., 0., 0., 1.]],

            [[1., 0., 0.,  ..., 0., 0., 0.],
             [1., 0., 1.,  ..., 1., 1., 1.],
             [1., 1., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 1., 1.],
             [1., 1., 0.,  ..., 0., 0., 1.],
             [1., 0., 0.,  ..., 1., 1., 0.]],

            [[1., 0., 1.,  ..., 1., 0., 0.],
             [1., 0., 0.,  ..., 0., 0., 1.],
             [0., 1., 1.,  ..., 1., 0., 0.],
             ...,
             [0., 0., 1.,  ..., 0., 1., 0.],
             [0., 0., 0.,  ..., 1., 0., 0.],
             [1., 0., 1.,  ..., 0., 0., 0.]]], device='cuda:0')
    """
    
    
    
    accept_logp    = accept_dist.log_prob(accept_samples)
    """
    accept_logp
    torch.Size([3, 96, 96])
    tensor([[[-0.8068, -1.1512, -0.5715,  ..., -0.5633, -0.6172, -0.4005],
             [-0.5521, -0.4408, -1.0158,  ..., -0.8487, -0.8781, -0.7325],
             [-0.3596, -1.0184, -0.4452,  ..., -0.6301, -0.7410, -0.6618],
             ...,
             [-0.7823, -0.4912, -0.4745,  ..., -0.5443, -0.9538, -1.0491],
             [-0.7614, -0.4733, -1.0259,  ..., -0.5022, -0.4641, -0.4426],
             [-0.5531, -0.4347, -0.9998,  ..., -0.4246, -0.4506, -1.0395]],

            [[-0.4859, -1.0432, -0.7347,  ..., -0.5499, -0.6647, -0.5707],
             [-0.5049, -1.0176, -0.5282,  ..., -0.7241, -0.7450, -1.0201],
             [-0.4131, -0.6019, -0.8594,  ..., -0.6961, -0.7247, -0.8017],
             ...,
             [-0.8991, -0.6220, -0.4136,  ..., -0.4746, -1.0442, -1.0968],
             [-0.6623, -0.6982, -0.3382,  ..., -0.3529, -0.3796, -1.1550],
             [-0.7779, -0.3291, -0.4245,  ..., -1.3286, -1.0327, -0.4774]],

            [[-0.1921, -0.7083, -0.4988,  ..., -0.6452, -0.9220, -0.6928],
             [-0.4239, -0.8227, -0.8028,  ..., -0.6913, -0.4863, -0.3608],
             [-0.5705, -0.6256, -0.5347,  ..., -0.6800, -0.7681, -0.7546],
             ...,
             [-0.6907, -0.4593, -1.0377,  ..., -0.5192, -0.9294, -0.5109],
             [-0.5256, -0.4000, -0.3762,  ..., -0.9759, -0.4415, -0.4411],
             [-0.8234, -0.5552, -1.0781,  ..., -0.4554, -0.4203, -0.4010]]],
           device='cuda:0', grad_fn=<NegBackward0>)
    """
    
    
    accept_mask    = accept_samples == 1.
    """
    accept_mask
    torch.Size([3, 96, 96])
    tensor([[[False, False,  True,  ...,  True,  True,  True],
             [ True,  True, False,  ..., False, False,  True],
             [ True, False,  True,  ...,  True, False,  True],
             ...,
             [False, False, False,  ..., False,  True,  True],
             [ True, False,  True,  ..., False, False, False],
             [False, False,  True,  ..., False, False,  True]],

            [[ True, False, False,  ..., False, False, False],
             [ True, False,  True,  ...,  True,  True,  True],
             [ True,  True, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False,  True,  True],
             [ True,  True, False,  ..., False, False,  True],
             [ True, False, False,  ...,  True,  True, False]],

            [[ True, False,  True,  ...,  True, False, False],
             [ True, False, False,  ..., False, False,  True],
             [False,  True,  True,  ...,  True, False, False],
             ...,
             [False, False,  True,  ..., False,  True, False],
             [False, False, False,  ...,  True, False, False],
             [ True, False,  True,  ..., False, False, False]]], device='cuda:0')
    """


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
        
        """
        heatmap.unfold(2, v, v) 
        torch.Size([3, 1, 96, 768, 8])
        tensor([[[[[-3.1139e-01,  4.6624e-01,  8.7289e-01,  ...,  2.0654e-01,5.5258e-01,  7.5843e-01],
           [-6.0347e-01,  6.6675e-02,  6.9844e-01,  ..., -3.6775e-01, -7.0950e-02,  1.6722e-01],
           [ 8.3655e-02,  7.3244e-01,  1.6664e+00,  ...,  9.3245e-01, 1.0129e+00,  1.0526e+00],
           ...,
           [ 4.0166e-02,  4.6767e-01,  7.8202e-01,  ...,  4.0522e-01, 2.4285e-01,  1.5102e-01],
           [-8.8576e-02,  6.8541e-02,  3.0978e-01,  ..., -2.0946e-02, -1.2726e-01, -1.8443e-01],
           [ 3.6482e-01,  4.2932e-01,  4.1033e-01,  ...,  2.1966e-01, 1.6747e-01,  1.0855e-01]],        
        """
  
        
        """
        heatmap.unfold(2, v, v).unfold(3, v, v) 
        torch.Size([3, 1, 96, 96, 8, 8])
        tensor([[[[[[-3.1139e-01, -6.0347e-01,  8.3655e-02,  ..., -2.3842e-01, -2.8693e-01, -1.3590e-01],
                    [ 4.6624e-01,  6.6675e-02,  7.3244e-01,  ...,  4.4079e-01, 4.7519e-01,  6.3573e-01],
                    [ 8.7289e-01,  6.9844e-01,  1.6664e+00,  ...,  1.0941e+00, 1.1578e+00,  1.2043e+00],
                    ...,
                    [ 2.0654e-01, -3.6775e-01,  9.3245e-01,  ...,  6.0666e-01, 6.7383e-01,  7.3352e-01],
                    [ 5.5258e-01, -7.0950e-02,  1.0129e+00,  ...,  6.9034e-01, 7.5798e-01,  8.7586e-01],
                    [ 7.5843e-01,  1.6722e-01,  1.0526e+00,  ...,  4.8296e-01, 6.0503e-01,  6.6835e-01]],

                   [[ 5.9942e-02,  1.5390e-01,  1.3652e-01,  ...,  7.3976e-02, 7.2214e-02,  6.4029e-02],
                    [ 8.4138e-01,  9.4949e-01,  9.7928e-01,  ...,  9.2576e-01, 9.0759e-01,  8.9840e-01],
                    [ 1.2997e+00,  1.3750e+00,  1.4528e+00,  ...,  1.3993e+00, 1.3908e+00,  1.4020e+00],
                    ...,
        """

        """
        heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v*v) 
        torch.Size([3, 1, 96, 96, 64])
        tensor([[[[[-3.1139e-01, -6.0347e-01,  8.3655e-02,  ...,  4.8296e-01, 6.0503e-01,  6.6835e-01],
                   [ 5.9942e-02,  1.5390e-01,  1.3652e-01,  ...,  6.1927e-01, 6.7140e-01,  7.0280e-01],
                   [ 7.9184e-02,  1.0621e-01,  1.6106e-01,  ...,  8.9157e-01, 9.1781e-01,  9.2466e-01],
                   ...,
                   [ 2.1664e-01,  1.9108e-01,  1.7046e-01,  ...,  2.6707e-01, 3.0826e-01,  3.3816e-01],
                   [ 1.7556e-01,  1.5038e-01,  1.3575e-01,  ...,  3.7192e-01, 2.4402e-01,  1.6215e-02],
                   [-1.0751e-01, -9.8728e-02, -8.1865e-02,  ...,  1.5102e-01, -1.8443e-01,  1.0855e-01]],

                  [[ 5.8723e-01, -7.2133e-02,  7.0270e-01,  ...,  1.9597e-01, 3.0520e-01,  2.9223e-01],
                   [ 3.7912e-01,  4.7910e-01,  4.5876e-01,  ...,  3.2231e-01, 3.7075e-01,  4.4711e-01],
                   [ 4.4098e-01,  4.8455e-01,  4.8971e-01,  ...,  6.5191e-01, 6.1169e-01,  5.7264e-01],
                   ...,
        """
        print("heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v*v)")
        print(heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v*v).size())

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
        """
        heatmap
        torch.Size([3, 1, 768, 768])        
        tensor([[[[-0.3114, -0.6035,  0.0837,  ...,  0.0402, -0.0886,  0.3648],
                  [ 0.4662,  0.0667,  0.7324,  ...,  0.4677,  0.0685,  0.4293],
                  [ 0.8729,  0.6984,  1.6664,  ...,  0.7820,  0.3098,  0.4103],
                  ...,
                  [-0.3958, -0.3238, -0.3147,  ..., -0.6240, -0.3621, -0.3230],
                  [-0.3962, -0.3367, -0.3223,  ..., -0.6545, -0.3331, -0.2514],
                  [-0.1963, -0.1049, -0.1933,  ..., -0.2999, -0.0940, -0.1392]]],


                [[[-0.2042, -0.4957, -0.1929,  ..., -0.4778, -0.4190,  0.1052],
                  [ 0.2604, -0.2550,  0.1652,  ..., -0.0804, -0.3094, -0.0021],
                  [ 0.5811,  0.2862,  0.8667,  ...,  0.3530,  0.0596,  0.0451],
                  ...,
                  [-0.3728, -0.2357, -0.2043,  ..., -0.5839, -0.3567, -0.3216],
                  [-0.4138, -0.3055, -0.2776,  ..., -0.6231, -0.3519, -0.2816],
                  [-0.2269, -0.1262, -0.2033,  ..., -0.3727, -0.2019, -0.2202]]],


                [[[-0.1296, -0.2416, -0.2713,  ...,  0.0927,  0.5003,  0.5097],
                  [ 0.2980,  0.2711,  0.4366,  ...,  0.5213,  0.6864,  0.6098],
                  [ 0.1327,  0.2894,  0.5958,  ...,  0.5674,  0.8474,  0.4908],
                  ...,
                  [-0.2740, -0.2406, -0.2267,  ..., -0.3642, -0.2350, -0.2399],
                  [-0.2637, -0.2226, -0.2063,  ..., -0.4063, -0.2329, -0.2066],
                  [-0.1747, -0.1406, -0.1976,  ..., -0.2057, -0.0943, -0.1603]]]],
               device='cuda:0', grad_fn=<SliceBackward0>)
        """

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        
        """
        heatmap_tiled
        torch.Size([3, 96, 96, 64])
        tensor([[[[-3.1139e-01, -6.0347e-01,  8.3655e-02,  ...,  4.8296e-01, 6.0503e-01,  6.6835e-01],
                  [ 5.9942e-02,  1.5390e-01,  1.3652e-01,  ...,  6.1927e-01, 6.7140e-01,  7.0280e-01],
                  [ 7.9184e-02,  1.0621e-01,  1.6106e-01,  ...,  8.9157e-01, 9.1781e-01,  9.2466e-01],
                  ...,
                  [ 2.1664e-01,  1.9108e-01,  1.7046e-01,  ...,  2.6707e-01, 3.0826e-01,  3.3816e-01],
                  [ 1.7556e-01,  1.5038e-01,  1.3575e-01,  ...,  3.7192e-01, 2.4402e-01,  1.6215e-02],
                  [-1.0751e-01, -9.8728e-02, -8.1865e-02,  ...,  1.5102e-01, -1.8443e-01,  1.0855e-01]],
        """

        
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
        )[::-1], dim=0).unsqueeze(0)
        
        """
        cgrid
        torch.Size([1, 2, 768, 768])
        tensor([[[[  0,   1,   2,  ..., 765, 766, 767],
                  [  0,   1,   2,  ..., 765, 766, 767],
                  [  0,   1,   2,  ..., 765, 766, 767],
                  ...,
                  [  0,   1,   2,  ..., 765, 766, 767],
                  [  0,   1,   2,  ..., 765, 766, 767],
                  [  0,   1,   2,  ..., 765, 766, 767]],

                 [[  0,   0,   0,  ...,   0,   0,   0],
                  [  1,   1,   1,  ...,   1,   1,   1],
                  [  2,   2,   2,  ...,   2,   2,   2],
                  ...,
                  [765, 765, 765,  ..., 765, 765, 765],
                  [766, 766, 766,  ..., 766, 766, 766],
                  [767, 767, 767,  ..., 767, 767, 767]]]], device='cuda:0')
        """

        
        cgrid_tiled = self._tile(cgrid)
        """
        cgrid_tiled
        tensor([[[[[  0,   1,   2,  ...,   5,   6,   7],
                   [  8,   9,  10,  ...,  13,  14,  15],
                   [ 16,  17,  18,  ...,  21,  22,  23],
                   ...,
                   [744, 745, 746,  ..., 749, 750, 751],
                   [752, 753, 754,  ..., 757, 758, 759],
                   [760, 761, 762,  ..., 765, 766, 767]],

                  [[  0,   1,   2,  ...,   5,   6,   7],
                   [  8,   9,  10,  ...,  13,  14,  15],
                   [ 16,  17,  18,  ...,  21,  22,  23],
                   ...,
                   [744, 745, 746,  ..., 749, 750, 751],
                   [752, 753, 754,  ..., 757, 758, 759],
                   [760, 761, 762,  ..., 765, 766, 767]],

        """
        
        print("cgrid_tiled")
        print(cgrid_tiled.size())

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
