from tensorflow.keras import backend as K


def get_single_output_size(y_pred):
    # Output consists of the outputs for both anchor, positive and negative input. Hence, the output for a single input
    # has the size "total output size / 3"
    return int(y_pred.shape[1] / 3)


def triplet_loss_l1(y_true, y_pred):
    single_output_size = get_single_output_size(y_pred)
    
    anchor_out = y_pred[:, 0:single_output_size]
    positive_out = y_pred[:, single_output_size:(2 * single_output_size)]
    negative_out = y_pred[:, (2 * single_output_size):(3 * single_output_size)]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))


def triplet_loss_l2(y_true, y_pred):
    single_output_size = get_single_output_size(y_pred)

    anchor_out = y_pred[:, 0:single_output_size]
    positive_out = y_pred[:, single_output_size:(2 * single_output_size)]
    negative_out = y_pred[:, (2 * single_output_size):(3 * single_output_size)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))


def triplet_loss_euler(y_true, y_pred):
    single_output_size = get_single_output_size(y_pred)

    anchor_out = y_pred[:, 0:single_output_size]
    positive_out = y_pred[:, single_output_size:(2 * single_output_size)]
    negative_out = y_pred[:, (2 * single_output_size):(3 * single_output_size)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))

    delta_plus = K.exp(pos_dist) / (K.exp(pos_dist) + K.exp(neg_dist))
    delta_min = K.exp(neg_dist) / (K.exp(pos_dist) + K.exp(neg_dist))

    return K.sqrt(K.sum(K.square(delta_plus + (delta_min - 1)), axis=-1)) ** 2


def triplet_loss_euler_2(y_true, y_pred):
    single_output_size = get_single_output_size(y_pred)

    anchor_out = y_pred[:, 0:single_output_size]
    positive_out = y_pred[:, single_output_size:(2 * single_output_size)]
    negative_out = y_pred[:, (2 * single_output_size):(3 * single_output_size)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = min(K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))
                   , K.sqrt(K.sum(K.square(positive_out - negative_out), axis=-1)))

    delta_plus = K.exp(pos_dist) / (K.exp(pos_dist) + K.exp(neg_dist))
    delta_min = K.exp(neg_dist) / (K.exp(pos_dist) + K.exp(neg_dist))

    return K.sqrt(K.sum(K.square(delta_plus + (1 - delta_min)), axis=-1)) ** 2