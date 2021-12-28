import argparse
import pickle
import time
from utils import Data, split_validation, translation
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--worker', type=int, default=3, help='number of worker')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0.0, help='l2 penalty')  #
parser.add_argument('--step', type=int, default=2, help='propogation steps')
parser.add_argument('--window', type=int, default=1, help='window size')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    llen = [len(train_data[0][i]) for i in range(len(train_data[0]))] + [len(test_data[0][i]) for i in range(len(test_data[0]))]
    print(max(llen),sum(llen)*1.0/len(llen))
    if not opt.dataset == 'diginetica':
        l = []
        for i in range(len(train_data[0])):
            l += list(train_data[0][i])
        l += list(train_data[1])

        for i in range(len(test_data[0])):
            l += list(test_data[0][i])
        l += list(test_data[1])
        l = set(l)
        print('total number of items', len(l))

        item_dic = {}
        for i in l:
           item_dic[i] = len(item_dic) + 1 #start from 1

        del l
        train_data = translation(train_data, item_dic)
        test_data = translation(test_data, item_dic)
        
        n_node = len(item_dic) + 1

    

    train_data = Data(train_data, opt.window)
    test_data = Data(test_data, opt.window)


    if opt.dataset == 'diginetica':
        n_node = 43098
    


    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0]
    bad_counter = 0




    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        train_model(model, train_data, opt)

        hit, mrr, hit10, mrr10 = test_model(model, test_data, opt)


        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_result[2] = hit10
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_result[3] = mrr10
            best_epoch[1] = epoch
            flag = 1

        print('Result:\n')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tHIT@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d\n'% (hit, mrr, hit10, mrr10, epoch))

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tHIT@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d\n'% (best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))




if __name__ == '__main__':
    main()
