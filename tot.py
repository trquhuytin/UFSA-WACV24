import numpy as np
import torch
import torch.nn.functional as F
from evaluation.grammar import Grammar
from evaluation.viterbi import Viterbi


def generate_optimal_transport(proto_scores, epsilon, p_gauss):
    """
    Function to generate the temporal optimal transport Q
    """
    q = proto_scores / epsilon
    q = torch.exp(q)
    if p_gauss is not None:
        q = q * torch.from_numpy(p_gauss).cuda()
    q = q.t()
    q = distributed_sinkhorn(q, 3)
    #q= torch.transpose(q, 0,1)
    return q


def distributed_sinkhorn(Q, nmb_iters):
    """
    Function used to perform distributed sinkhorn algorithm for temporal optimal transport
    """
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        curr_sum = torch.sum(Q, dim=1)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def get_cost_matrix(batch_size, num_videos, num_clusters, sigma):
    """
    Returns
    ----
    Cost matrix for num_videos (where all values all concentrated near the diagonal)
    """
    cost_matrix = generate_matrix(int(batch_size / num_videos), num_clusters)
    cost_matrix = np.vstack([cost_matrix] * num_videos)
    p_gauss = gaussian(cost_matrix, sigma=sigma)

    return p_gauss

def get_complete_cost_matrix(vid_len, num_videos, num_clusters, sigma):
    """
    Returns
    ----
    Cost matrix for num_videos (where all values all concentrated near the diagonal)
    """
    cost_matrix = generate_matrix(vid_len, num_clusters)
    cost_matrix = np.vstack([cost_matrix] * num_videos)
    p_gauss = gaussian(cost_matrix, sigma=sigma)

    return p_gauss


def gaussian(cost, sigma):
    """
    This functions returns the prior distribution Tij as mentioned in the paper
    """
    return (1 / (sigma * 2 * 3.142)) * (np.exp(-cost / (2 * (sigma ** 2))))


def generate_matrix(num_elements, num_clusters):
    """
    Generates dij as mentioned in the paper where dij is the distance entry(i,j)
    to the diagonal line
    """
    cost_matrix = np.zeros((num_elements, num_clusters))

    n = num_elements
    m = num_clusters

    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = ((abs(i / n - j / m)) / (np.sqrt((1 / n ** 2) + (1 / m ** 2)))) ** 2

    return cost_matrix


def get_proto_labels(embeddings, prototypes):
    prob_scores = np.matmul(embeddings, np.transpose(prototypes))
    assignments = np.argmax(prob_scores, axis=1)
    assignments = assignments
    return assignments


def get_proto_likelihood(features, prototypes):
    scores = np.matmul(features, np.transpose(prototypes))
    probs = softmax(scores / 0.1)
    probs = np.clip(probs, 1e-30, 1)
    return np.log(probs)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 

def viterbi_inner(likelihood_grid, n_frames, pi):
    grammar = Grammar(pi)
    viterbi = Viterbi(grammar=grammar, probs=(-1 * likelihood_grid))
    viterbi.inference()
    viterbi.backward(strict=True)
    z = np.ones(n_frames, dtype=int) * -1
    z = viterbi.alignment()
    score = viterbi.loglikelyhood()
    return z, score
