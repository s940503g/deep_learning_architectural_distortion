from keras import backend as K

def precision(y_true, y_pred):
    tp = K.sum(K.round(y_true * y_pred))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    precision = tp / (tp + fp + K.epsilon())
    return precision

def recall(y_true, y_pred):
    tp = K.sum(K.round(y_true * y_pred)) 
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    recall = tp / (tp + fn + K.epsilon())
    return recall

def specificity(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg) / K.sum(y_neg) + K.epsilon()
    return tn

def F1(y_true, y_pred):
    beta = 1
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    return (beta*beta + 1)*p*r / (beta*beta*p + r + K.epsilon())
