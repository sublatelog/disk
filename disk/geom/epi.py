import torch

from torch_dimcheck import dimchecked

# reward > classify > asymmdist_from_imgs > ims2F > ims2E > cross_product_matrix
# 行列積
@dimchecked
def cross_product_matrix(v: [3]) -> [3, 3]:
    ''' following
        en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    '''

    # V
    # tensor([0.3014, 1.1819, 7.4409], device='cuda:0')
    
    v2 = torch.tensor([
                            [    0, -v[2],  v[1]],
                            [ v[2],     0, -v[0]],
                            [-v[1],  v[0],     0]
                        ], dtype=v.dtype, device=v.device)
    """
    v2
    tensor([[ 0.0000, -7.4409,  1.1819],
            [ 7.4409,  0.0000, -0.3014],
            [-1.1819,  0.3014,  0.0000]], device='cuda:0')
    """
    
    return torch.tensor([
                            [    0, -v[2],  v[1]],
                            [ v[2],     0, -v[0]],
                            [-v[1],  v[0],     0]
                        ], dtype=v.dtype, device=v.device)

@dimchecked
def xy_to_xyw(xy: [2, 'N']) -> [3, 'N']:
    ones = torch.ones(1, xy.shape[1], device=xy.device, dtype=xy.dtype) # 全て1の列を右端に追加
    return torch.cat([xy, ones], dim=0)


# reward > classify > asymmdist_from_imgs > ims2F > ims2E
@dimchecked
def ims2E(im1, im2) -> [3, 3]:
    
    """
            
    im1.R
    tensor([[-0.9271, -0.0076, -0.3747],
            [ 0.0481,  0.9891, -0.1390],
            [ 0.3716, -0.1469, -0.9167]], device='cuda:0')
            
    im2.R
    tensor([[-0.9289, -0.0248, -0.3694],
            [ 0.1506,  0.8862, -0.4382],
            [ 0.3383, -0.4627, -0.8194]], device='cuda:0')
            
    im1.R.T        
    tensor([[-0.9271,  0.0481,  0.3716],
            [-0.0076,  0.9891, -0.1469],
            [-0.3747, -0.1390, -0.9167]], device='cuda:0')
            
    im2.R @ im1.R.T 
    1行目x1列目、2行目x1列目、3行目x1列目
    1行目x2列目、2行目x2列目、3行目x2列目
    
    tensor([[ 0.9998, -0.0178, -0.0029],
            [ 0.0178,  0.9447,  0.3275],
            [-0.0031, -0.3275,  0.9448]], device='cuda:0')
    """
    
    # im1の回転ベクトルをim2の回転ベクトルで回転させる
    R = im2.R @ im1.R.T # img2.rotation x img1.rotation.T
    
    # im2と回転させたim1の差を取る
    T = im2.T - R @ im1.T
    
    # 行列積(3,3)を回転に適用
    return cross_product_matrix(T) @ R


# reward > classify > asymmdist_from_imgs > ims2F
@dimchecked
def ims2F(im1, im2) -> [3, 3]:
    
    # E=行列積(3,3)を回転に適用
    E = ims2E(im1, im2)
    
    return im2.K_inv.T @ E @ im1.K_inv


@dimchecked
def symdimm(x1: [2, 'N'], x2: [2, 'M'], im1, im2) -> ['N', 'M']:
    x1n = im1.K_inv @ xy_to_xyw(x1)
    x2n = im2.K_inv @ xy_to_xyw(x2)

    E = ims2E(im1, im2)

    E_x1  = E @ x1n
    Et_x2 = E.T @ x2n
    x2_E_x1 = x2n.T @ E_x1

    n = lambda v: torch.norm(v, p=2, dim=0)

    n1 = 1 / n(E_x1[:2])[None, :]
    n2 = 1 / n(Et_x2[:2])[:, None]
    norm = n1 + n2
    dist = x2_E_x1.pow(2) * norm
    return dist.T


# reward > classify > asymmdist_from_imgs > ims2F > ims2E > cross_product_matrix > asymmdist
@dimchecked
def asymmdist(x1: [2, 'N'], x2: [2, 'M'], F: [3, 3]) -> ['N', 'M']:
    '''
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    '''

    x1_h = xy_to_xyw(x1)
    x2_h = xy_to_xyw(x2)
    
    
    print("x2_h")
    print(x2_h)
    
    """
    F
    tensor([[-2.7759e-09,  1.7227e-07, -8.7349e-05],
            [-1.9068e-07,  3.5481e-09,  3.7626e-04],
            [ 5.6962e-05, -2.1005e-04,  2.4813e-04]], device='cuda:0')
    """    

    Ft_x2 = F.T @ x2_h # x2にFを適用
    
    """
    Ft_x2
    tensor([[ 5.6948e-05,  5.5950e-05,  5.6486e-05,  ..., -9.0157e-05, -9.1351e-05, -9.1171e-05],
            [-2.0919e-04, -2.0642e-04, -2.0419e-04,  ..., -8.2458e-05, -7.9336e-05, -7.8651e-05],
            [-1.8862e-04,  2.9509e-04, -1.9692e-03,  ...,  2.2325e-01, 2.2394e-01,  2.2321e-01]], device='cuda:0')
    """
    
    norm  = torch.norm(Ft_x2[:2], p=2, dim=0) # 正規化
    
    """
    norm
    tensor([0.0002, 0.0002, 0.0002,  ..., 0.0001, 0.0001, 0.0001], device='cuda:0')
    """
    
    
    dist  = (Ft_x2 / norm).T @ x1_h # Fを適用したx2を正規化してx1に適用
    return dist.T

# reward > classify > asymmdist_from_imgs
@dimchecked
def asymmdist_from_imgs(x1: [2, 'N'], x2: [2, 'M'], im1, im2) -> ['N', 'M']:
    
    """
    im1, im2
    R = im2.R @ im1.R.T # img2.rotation x img1.rotation.T
    T = im2.T - R @ im1.T
    cross_product_matrix = torch.tensor([
                                        [    0, -T[2],  T[1]],
                                        [ T[2],     0, -T[0]],
                                        [-T[1],  T[0],     0]
                                        ])
                        
    E = cross_product_matrix @ R
    F = im2.K_inv.T @ E @ im1.K_inv
     
    x1_h = xy_to_xyw(x1) # 全て1の列を右端に追加
    x2_h = xy_to_xyw(x2) # 全て1の列を右端に追加

    Ft_x2 = F.T @ x2_h　# x2にFを適用
    norm  = torch.norm(Ft_x2[:2], p=2, dim=0) # 正規化
    dist  = (Ft_x2 / norm).T @ x1_h # Fを適用したx2を正規化してx1に適用
    
    return dist.T
    """
    
    F = ims2F(im1, im2)
    return asymmdist(x1, x2, F)


@dimchecked
def p_asymmdist(x1: [2, 'N'], x2: [2, 'N'], F: [3, 3]) -> ['N']:
    '''
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    '''

    x1_h = xy_to_xyw(x1)
    x2_h = xy_to_xyw(x2)

    Ft_x2 = F.T @ x2_h
    norm  = torch.norm(Ft_x2[:2], p=2, dim=0)
    Ft_x2_n = Ft_x2 / norm

    return torch.einsum('ca,ca->a', (Ft_x2_n, x1_h))

@dimchecked
def p_asymmdist_from_imgs(x1: [2, 'N'], x2: [2, 'N'], im1, im2) -> ['N']:
    F = ims2F(im1, im2)
    return p_asymmdist(x1, x2, F)
