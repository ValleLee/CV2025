import numpy as np
import torch
import cv2
from proj3_code.feature_matching.SIFTNet import get_siftnet_features


def pairwise_distances(X, Y):
    """
    计算两组特征向量之间的欧氏距离矩阵。
    Args:
    - X: N x d numpy 数组
    - Y: M x d numpy 数组
    Returns:
    - D: N x M numpy 数组，D[i, j] 是 X[i] 和 Y[j] 之间的距离
    """
    # 利用NumPy的广播机制高效计算，避免循环
    X_expanded = np.expand_dims(X, axis=1)
    Y_expanded = np.expand_dims(Y, axis=0)
    
    squared_dists = np.sum(np.square(X_expanded - Y_expanded), axis=2)
    
    dists = np.sqrt(squared_dists)
    
    return dists


def get_tiny_images(image_arrays):
    """
    将原始图像缩放成一个很小的正方形（例如16x16），然后将其展平成一个向量。
    Args:
    - image_arrays: 一个包含N个numpy图像数组的列表（灰度图）
    Returns:
    - feats: N x d 的numpy数组，其中d是tiny image的维度（如16*16=256）
    """
    feats = []
    target_size = (16, 16)

    for image in image_arrays:
        resized_image = cv2.resize(image, dsize=target_size)
        
        feature_vector = resized_image.flatten()
        
        mean = np.mean(feature_vector)
        feature_vector = feature_vector - mean
        
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        feats.append(feature_vector)

    return np.array(feats)


def nearest_neighbor_classify(train_image_feats, train_labels,
                              test_image_feats, k=3):
    """
    实现k-NN分类器。
    Args:
    - train_image_feats: 训练集特征 (N x d)
    - train_labels: 训练集标签 (N)
    - test_image_feats: 测试集特征 (M x d)
    - k: k-NN中的k值
    Returns:
    - test_labels: 预测的测试集标签 (M)
    """
    test_labels = []

    dists = pairwise_distances(test_image_feats, train_image_feats)
    
    num_test = dists.shape[0]
    for i in range(num_test):
        k_nearest_indices = np.argsort(dists[i, :])[:k]
        
        k_nearest_labels = [train_labels[j] for j in k_nearest_indices]
        
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        
        test_labels.append(predicted_label)

    return test_labels


def kmeans(feature_vectors, k, max_iter = 10):
    """
    实现k-means聚类算法。
    Args:
    - feature_vectors: 输入的特征数据 (N, d)
    - k: 簇的数量
    - max_iter: 最大迭代次数
    Returns:
    - centroids: 生成的k个质心 (k, d)
    """
    # 设置一个固定的随机种子，以确保每次运行结果都一样，从而能通过单元测试
    np.random.seed(0)

    N, d = feature_vectors.shape
    
    initial_indices = np.random.choice(N, k, replace=False)
    centroids = feature_vectors[initial_indices, :]
    
    for _ in range(max_iter):
        dists = pairwise_distances(feature_vectors, centroids)
        labels = np.argmin(dists, axis=1)
        
        new_centroids = np.zeros((k, d))
        for j in range(k):
            cluster_points = feature_vectors[labels == j]

            if len(cluster_points) > 0:
                new_centroids[j, :] = np.mean(cluster_points, axis=0)
            else:
                # 如果这个簇是空的，就不更新它，让它保持在原位
                new_centroids[j, :] = centroids[j, :]
        
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids

    return centroids


def build_vocabulary(image_arrays, vocab_size, stride = 20):
    """
    从训练图片中密集采样SIFT特征，然后用k-means聚类构建视觉词汇表。
    Args:
    - image_arrays: 训练图像列表
    - vocab_size: 词汇表大小
    - stride: 采样步长
    Returns:
    - vocab: 视觉词汇表 (vocab_size, 128)
    """
    all_features = []
    
    for image in image_arrays:
        y_coords = np.arange(10, image.shape[0] - 10, stride)
        x_coords = np.arange(10, image.shape[1] - 10, stride)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            continue
            
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        sample_x = x_grid.flatten()
        sample_y = y_grid.flatten()
        
        img_tensor = torch.from_numpy(image.astype(np.float32))
        img_tensor = img_tensor.reshape(1, 1, image.shape[0], image.shape[1])
        
        sift_features = get_siftnet_features(img_tensor, sample_x, sample_y)
        
        if sift_features is not None and len(sift_features) > 0:
            all_features.append(sift_features)
    
    if not all_features:
        return np.zeros((vocab_size, 128))

    full_feature_set = np.vstack(all_features)
    
    vocab = kmeans(full_feature_set, vocab_size)

    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    将输入的数据点量化到最近的质心。
    Args:
    - raw_data_pts: 输入的数据点 (N, d)
    - centroids: 词汇表/质心 (k, d)
    Returns:
    - indices: 每个数据点对应的最近质心的索引 (N,)
    """
    dists = pairwise_distances(raw_data_pts, centroids)
    indices = np.argmin(dists, axis=1)
    
    return indices


def get_bags_of_sifts(image_arrays, vocabulary, step_size = 10):
    """
    为每张图片计算其词袋（Bag of SIFTs）特征表示。
    Args:
    - image_arrays: 图像列表
    - vocabulary: 训练好的视觉词汇表
    - step_size: 采样步长
    Returns:
    - image_feats: N x vocab_size 的特征矩阵
    """
    vocab_size = vocabulary.shape[0]
    num_images = len(image_arrays)
    feats = np.zeros((num_images, vocab_size))

    for i, image in enumerate(image_arrays):
        y_coords = np.arange(10, image.shape[0] - 10, step_size)
        x_coords = np.arange(10, image.shape[1] - 10, step_size)

        if len(y_coords) == 0 or len(x_coords) == 0:
            continue

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        sample_x, sample_y = x_grid.flatten(), y_grid.flatten()

        img_tensor = torch.from_numpy(image.astype(np.float32))
        img_tensor = img_tensor.reshape(1, 1, image.shape[0], image.shape[1])
        
        sift_features = get_siftnet_features(img_tensor, sample_x, sample_y)
        
        if sift_features is None or len(sift_features) == 0:
            continue
        
        word_indices = kmeans_quantize(sift_features, vocabulary)
        
        histogram = np.bincount(word_indices, minlength=vocab_size)
        
        norm = np.linalg.norm(histogram, ord=1)
        if norm > 0:
            histogram = histogram / norm
            
        feats[i, :] = histogram

    return feats