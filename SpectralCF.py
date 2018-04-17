import random
import multiprocessing
cores = multiprocessing.cpu_count()
from SpectralCF import *

from load_data import *


MODEL = 'GraphCF'
DATASET = 'ml-1m'

EMB_DIM = 16
BATCH_SIZE = 1024
DECAY = 0.001
LAMDA = 1
K = 3
N_EPOCH = 200
LR = 0.001
DROPOUT = 0.0
#NEGATIVE_SIZE = 3

DIR = 'data/'+DATASET+'/'



data_generator = Data(train_file=DIR+'train_users.dat', test_file=DIR+'test_users.dat', batch_size=BATCH_SIZE)
USER_NUM, ITEM_NUM = data_generator.get_num_users_items()


def simple_test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    training_items = data_generator.train_items[u]
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)




    recall_20 = ut.recall_at_k(r, 20, len(user_pos_test))
    recall_40 = ut.recall_at_k(r, 40, len(user_pos_test))
    recall_60 = ut.recall_at_k(r, 60, len(user_pos_test))
    recall_80 = ut.recall_at_k(r, 80, len(user_pos_test))
    recall_100 = ut.recall_at_k(r, 100, len(user_pos_test))

    ap_20 = ut.average_precision(r,20)
    ap_40 = ut.average_precision(r, 40)
    ap_60 = ut.average_precision(r, 60)
    ap_80 = ut.average_precision(r, 80)
    ap_100 = ut.average_precision(r, 100)


    return np.array([recall_20,recall_40,recall_60,recall_80,recall_100, ap_20,ap_40,ap_60,ap_80,ap_100])


def simple_test(sess, model, users_to_test):
    result = np.array([0.] * 10)
    pool = multiprocessing.Pool(cores)
    batch_size = BATCH_SIZE
    #all users needed to test
    test_users = users_to_test
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        FLAG = False
        if len(user_batch) < batch_size:
            user_batch += [user_batch[-1]] * (batch_size - len(user_batch))
            user_batch_len = len(user_batch)
            FLAG = True
        user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)

        if FLAG == True:
            batch_result = batch_result[:user_batch_len]
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def main():


    model = SpectralCF(K=K, graph=data_generator.R, n_users=USER_NUM, n_items=ITEM_NUM, emb_dim=EMB_DIM,
                     lr=LR, decay=DECAY, batch_size=BATCH_SIZE,DIR=DIR)
    print(model.model_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    best = None
    for epoch in range(N_EPOCH):
        users, pos_items, neg_items = data_generator.sample()
        _, loss = sess.run([model.updates, model.loss],
                                           feed_dict={model.users: users, model.pos_items: pos_items,
                                                      model.neg_items: neg_items})

        users_to_test = list(data_generator.test_set.keys())

        ret = simple_test(sess, model, users_to_test)



        print('Epoch %d training loss %f' % (epoch, loss))
        print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
              % (ret[0],ret[1],ret[2],ret[3],ret[4]))
        print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
              % (ret[5], ret[6], ret[7], ret[8], ret[9]))


if __name__ == '__main__':
    main()
