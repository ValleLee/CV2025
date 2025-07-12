import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """
    # 利用Numpy的广播机制，高效地计算两个特征集之间的欧氏距离矩阵
    # (a-b)^2 = a^2 - 2ab + b^2
    
    # 计算features1中每个特征的L2范数的平方
    f1_norm_sq = np.sum(np.square(features1), axis=1, keepdims=True) # 形状 (n, 1)
    
    # 计算features2中每个特征的L2范数的平方
    f2_norm_sq = np.sum(np.square(features2), axis=1, keepdims=True) # 形状 (m, 1)

    # 计算两个特征集的点积
    dot_product = features1 @ features2.T # 形状 (n, m)

    # 利用广播机制计算距离的平方
    # dists_sq[i,j] = f1_norm_sq[i] - 2*dot_product[i,j] + f2_norm_sq[j]
    dists_sq = f1_norm_sq - 2 * dot_product + f2_norm_sq.T
    
    # 对负值进行处理（由于数值计算误差可能出现极小的负值），然后开方
    dists = np.sqrt(np.maximum(0, dists_sq))
    
    return dists



def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is
      an index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    # 1. 计算两组特征之间的距离矩阵
    dists = compute_feature_distances(features1, features2)

    # 2. 对距离矩阵的每一行进行排序，找到每个feature1的最近邻和次近邻
    # np.argsort返回的是排序后的原始索引
    sorted_indices = np.argsort(dists, axis=1)
    
    # 最近邻的索引和距离
    nn1_indices = sorted_indices[:, 0]
    nn1_dists = dists[np.arange(dists.shape[0]), nn1_indices]
    
    # 次近邻的索引和距离
    nn2_indices = sorted_indices[:, 1]
    nn2_dists = dists[np.arange(dists.shape[0]), nn2_indices]

    # 3. Lowe's Ratio Test: 只有当最近邻距离与次近邻距离之比小于一个阈值时，才接受该匹配
    # 阈值通常取0.7到0.8之间，这里我们选择0.8
    ratio_threshold = 0.8
    # 避免除以零的情况
    ratios = nn1_dists / (nn2_dists + 1e-6)
    
    # 找到所有通过了比例测试的匹配
    passed_indices = np.where(ratios < ratio_threshold)[0]

    # 4. 整理输出结果
    # 匹配对的索引
    matches = np.stack((passed_indices, nn1_indices[passed_indices]), axis=1)
    # 置信度可以是1减去比例，比例越小，置信度越高
    confidences = 1 - ratios[passed_indices]

    return matches, confidences