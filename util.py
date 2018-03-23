import numpy as np
import scipy.sparse as sp
import torch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel):
    resultsString = ''
    s = '-- Models ranked by accuracy --'
    print(s)
    resultsString += s + "\n"
    i = 1
    for k in sorted(accsPerModel, key=accsPerModel.get)[::-1]:
        s = '#%d: model%f acc %f' % (i, k, accsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    i = 1
    s = '-- Models ranked by size --'
    print(s)
    resultsString += s + "\n"
    for k in sorted(paramsPerModel, key=paramsPerModel.get):
        s = '#%d: model%f size %d' % (i, k, paramsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    i = 1
    for k in sorted(rewardsPerModel, key=rewardsPerModel.get)[::-1]:
        s = '#%d: model%f reward %f ' % (i, k, rewardsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    if resultsFile:
        resultsFile.write(resultsString)