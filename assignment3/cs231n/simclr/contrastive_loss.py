import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    numerator = torch.dot(z_i, z_j)
    denominator = torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
    # the default form of torch.linalg.norm is Frobenius Norm. e.g. torch.linalg.norm(z_i) = ||z_i||.
    
    norm_dot_product = numerator / denominator
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        # where z_k is the left output of MLP in the positive pair, while z_k_N the right output
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        numerator = torch.exp(sim(z_k, z_k_N) / tau)
        denominator_k = 0
        denominator_k_N = 0
        
        for i in range(2*N): 
            z_i = out[i]
            if i != k: 
                denominator_k += torch.exp(sim(z_k, z_i) / tau)
            if i != (k+N): 
                denominator_k_N += torch.exp(sim(z_k_N, z_i) / tau)
        total_loss += -torch.log(numerator / denominator_k)
        total_loss += -torch.log(numerator / denominator_k_N)
        
        
        # numerator = torch.exp(sim(z_k, z_k_N) / tau)
        # denominator_k = 0
        # for i in range(2*N): 
        #     z_i = out[i]
        #     if i != k: 
        #         denominator_k += torch.exp(sim(z_k, z_i) / tau)
        # total_loss += -torch.log(numerator / denominator_k)

        # numerator = torch.exp(sim(z_k_N, z_k) / tau)    # note sim(z_k, z_k_N) equals sim(z_k_N, z_k), so actually we don't have to count it again
        # denominator_k_N = 0
        # for i in range(2*N): 
        #     z_i = out[i]
        #     if i != (k+N): 
        #         denominator_k_N += torch.exp(sim(z_k_N, z_i) / tau)
        # total_loss += -torch.log(numerator / denominator_k_N)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # copied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    denominator = torch.linalg.norm(out_left, dim=1) * torch.linalg.norm(out_right, dim=1)
    pos_pairs = (torch.diag(out_left.mm(out_right.T)) / denominator).unsqueeze(1)
    
    # 1. torch.linalg.norm: Computes a vector or matrix norm.
    # Note: linalg is short for linear algebra. 
    
    # 2.
    # torch.diag: 
    #     If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
    #     If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input.
    
    # 3. 
    # torch.unsqueeze(): Returns a new tensor with a dimension of size one inserted at the specified position.
    # torch.squeeze: Returns a tensor with all specified dimensions of input of size 1 removed.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    denominator = torch.linalg.norm(out, dim=1).outer(torch.linalg.norm(out, dim=1))
    sim_matrix = out.mm(out.T) / denominator

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = None
    
    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    exponential = torch.exp(sim_matrix / tau)
    # The assert keyword lets you test if a condition in your code returns True, if not, the program will raise an AssertionError.
    assert exponential.shape == (2 * N, 2 * N)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = None
    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    denom = torch.sum(exponential, dim=1).unsqueeze(1)
    assert denom.shape == (2 * N, 1)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    similarity = torch.cat((sim_positive_pairs(out_left, out_right), sim_positive_pairs(out_right, out_left)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py
    numerator = torch.exp(similarity / tau)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # cpoied from: https://github.com/samlkrystof/Stanford-CS231n/blob/main/assignment3/cs231n/simclr/contrastive_loss.py 
    loss = -torch.sum(torch.log(numerator / denom)) / (2 * N)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))