import numpy as np

def fixed_range_with_infinite_tail_score(X, K, global_min=0, global_max=600, sigma_scale=0.5):
    X = np.array(X)
    N = len(X)
    width = (global_max - global_min) / (K - 1)
    sigma = width * sigma_scale

    scores = np.zeros((K, N))

    for i in range(K):
        if i < K - 1:
            a = global_min + i * width
            b = a + width
            center = (a + b) / 2
            for j, x in enumerate(X):
                if a <= x <= b:
                    score = 1.0
                else:
                    score = np.exp(- ((x - center) ** 2) / (2 * sigma ** 2))
                scores[i, j] = score
        else:
            # 最后一组：x > global_max
            a = global_max
            center = global_max + width / 2
            for j, x in enumerate(X):
                if x > a:
                    score = 1.0
                else:
                    score = np.exp(- ((x - center) ** 2) / (2 * sigma ** 2))
                scores[i, j] = score

    return scores


X = [0, 0, 0 ,100, 250, 400, 600, 620, 680, 750, 2000]
scores = fixed_range_with_infinite_tail_score(X, K=5, global_min=0, global_max=600, sigma_scale=7)
print(np.round(scores, 2))
