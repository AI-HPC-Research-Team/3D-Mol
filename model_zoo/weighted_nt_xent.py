import paddle
import numpy as np


def _get_correlated_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    a = (diag + l1 + l2)
    a = 1 - a
    mask = paddle.to_tensor(a, dtype=paddle.bool)
    #    mask = (1 - mask).type(paddle.bool)
    return mask


def _dot_simililarity(x, y):
    v = paddle.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def _cosine_simililarity(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    _cosine_similarity = paddle.nn.CosineSimilarity(axis=-1)
    v = _cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
    return v


def WeightedNTXentLoss_func(x1, x2, fp_score, temperature=0.1, use_cosine_similarity=True, lambda_1=0.5, rms=None):
    if use_cosine_similarity:
        similarity_function = _cosine_simililarity
    else:
        similarity_function = _dot_simililarity
    assert x1.size == x2.size
    batch_size = x1.shape[0]

    fp_score = 1 - lambda_1 * paddle.to_tensor(fp_score, dtype=paddle.float32)
    fp_score = paddle.concat([fp_score, fp_score], axis=0)
    fp_score = paddle.concat([fp_score, fp_score], axis=-1)
    representations = paddle.concat([x2, x1], axis=0)
    similarity_matrix = similarity_function(representations, representations)
    l_pos = paddle.diag(similarity_matrix, batch_size)


    r_pos = paddle.diag(similarity_matrix, -batch_size)
    positives = paddle.concat([l_pos, r_pos]).reshape([2 * batch_size, 1])
    mask_samples_from_same_repr = _get_correlated_mask(batch_size)
    negatives = similarity_matrix[mask_samples_from_same_repr].reshape([2 * batch_size, -1])
    negatives *= fp_score
    rms = paddle.unsqueeze(paddle.concat([rms, rms], axis=0), axis=-1)
    positives *= rms
    logits = paddle.concat((positives, negatives), axis=1)
    logits /= temperature
    labels = paddle.zeros([2 * batch_size], dtype="int64")
    criterion = paddle.nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    return loss / (2 * batch_size)

