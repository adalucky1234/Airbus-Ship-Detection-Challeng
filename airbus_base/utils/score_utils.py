import numpy as np


def IoU(preds, true):
    """ Base IOU score evaluating function
    Args:
        preds - batches of predictions
        true - batches of ground truth

    Returns:
        if preds and true are (N*H*W) tensor, a (N,) array of scores
    """
    preds = preds.astype(bool)
    true = true.astype(bool)

    assert(preds.shape == true.shape)
    num_axes = len(preds.shape)

    Intersection = (preds*true).astype(int)
    Union = (preds+true).astype(int)

    for i in range(num_axes-1):
        Intersection = np.sum(Intersection, axis=1)
        Union = np.sum(Union, axis=1)

    return np.divide(Intersection.astype(float),
                     Union,
                     out=np.ones(shape=Intersection.shape),
                     where=Union != 0)

def F2_score(preds, true, Beta=2):
    empty_preds = GetEmptyMasks(preds)
    empty_true = GetEmptyMasks(true)
    FP = np.sum(empty_true*np.invert(empty_preds))
    FN = np.sum(np.invert(empty_true)*empty_preds)

    thresholds = np.arange(0.5, 1, 0.05)
    batch_size = int((preds.shape)[0])
    num_thresholds = int((thresholds.shape)[0])
    TP_mat = np.zeros((batch_size,batch_size)).astype(int)
    for i,t in enumerate(thresholds):
        TP_mat[:,i] = (raw_IoU>t).astype(int)
    TP = np.sum(TP_mat, axis=-1)

    
    F2 = np.divide((1+Beta*Beta)*TP,((1+Beta*Beta)*TP+Beta*Beta*FP+FN))
    return F2


def GetEmptyMasks(masks):
    num_axes = len(masks.shape)
    sum = masks
    for i in range(num_axes-1):
        sum= np.sum(sum, axis=1)

    return sum == 0
