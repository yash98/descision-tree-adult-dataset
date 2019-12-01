import numpy as np
import sys


def entropy2(a, b):
    sum_ab = a + b
    if a == sum_ab or b == sum_ab:
        return 0

    p1 = a/sum_ab
    p2 = b/sum_ab
    return -1*(p1*np.log2(p1)+p2*np.log2(p2))


def entropy_arr(npa):
    entr_l = np.zeros(npa.shape[0])
    for i in range(npa.shape[0]):
        entr_l[i] = entropy2(npa[i][0], npa[i][1])
    return entr_l
    # npa1 = npa[:, 0]
    # npa2 = npa[:, 1]
    # npa_sum = npa1+npa2
    # npa1 = npa1/npa_sum
    # npa2 = npa2/npa_sum
    # npa1 = np.clip(npa1, a_min=0.0000001, a_max=1)
    # npa2 = np.clip(npa2, a_min=0.0000001, a_max=1)
    # r = -1*(npa1*np.log2(npa1)+npa2*np.log2(npa2))
    # r = np.clip(r, a_min=0, a_max=1)
    # return r


class Dataset:

    def __init__(self, data_filename, val_type_list, quantity_type_list):
        """
        data_filename: string - filename of datafile to be converted to Dataset D.S
        val_type_list: dtype formats for numpy.loadtxt
        quantity_type_list: list of chars 'C' or 'D' denoting Continuous or Discrete
        """

        self.num_features = len(quantity_type_list) - 1
        data_file = open(data_filename, 'r')

        self.column_title = data_file.readline().split(', ')
        self.column_title[-1] = self.column_title[-1].strip(
            '\n').strip('?')

        self.quantity_type_list = quantity_type_list
        self.val_type_list = val_type_list
        self.data_array = np.loadtxt(data_file, delimiter=', ', dtype='S10')
        # self.data_read = np.loadtxt(data_file, delimiter=', ', dtype={'names': tuple(
        #     self.column_title), 'formats': tuple(val_type_list)})
        print("loading done " + data_filename)

        # if training:
        #     self.encode()
        #     print("encoding done " + data_filename)
        # else:
        #     print("skipping encoding")

    # def encode(self):
    #     self.encoding = [dict() for _ in range(self.num_features)]
    #     self.rev_encoding = [dict() for _ in range(self.num_features)]
    #     self.max_variation = np.zeros(self.num_features, dtype=int)
    #     self.data_array = np.zeros((self.data_read.shape[0], self.num_features+1), dtype=int)

    #     for i in range(len(self.data_read)):
    #         for j in range(self.num_features):
    #             current = self.data_read[i][j]
    #             if self.val_type_list[j] != 'i4':
    #                 if current in self.encoding:
    #                     self.data_array[i][j] = self.encoding[j][current]
    #                 else:
    #                     self.encoding[j][current] = self.max_variation[j]
    #                     self.rev_encoding[j][self.max_variation[j]] = current
    #                     self.max_variation[j] += 1
    #                     self.data_array[i][j] = self.encoding[j][current]
    #             else:
    #                 self.data_array[i][j] = current
    #         self.data_array[i][-1] = self.data_read[i][-1]


class Node:

    def __init__(self, type_n, name, entropy=None, data_array=None, remaining_impure_features=None):
        """
        type_n: 'C' | 'D' | 'N' | 'U' -> Continuous | Discrete | No more split | Untrained
        """

        assert(type_n == 'C' or type_n == 'D' or type_n == 'N' or type_n == 'U')
        self.type_n = type_n
        self.name = name

        # training time vars, to be cleared later
        self.entropy = entropy
        self.data_array = data_array
        self.remaining_impure_features = remaining_impure_features

        if type_n == 'C':
            self.children = []
            self.edge_label = []
            self.column_id = -1
        elif type_n == 'D':
            self.children = []
            self.median = None
            self.column_id = -1
        elif type_n == 'N':
            self.pred = 0
        else:
            pass

    def set_entropy(self):
        classes = [0, 0]
        #  FIXME: optimize using np fn
        for row in self.data_array:
            classes[int(row[-1])] += 1
        self.entropy = entropy2(classes[0], classes[1])

    def clear_train_garbage(self):
        # del(self.entropy)
        del(self.data_array)
        del(self.remaining_impure_features)


class Decision_Tree:
    def __init__(self, dataset):
        self.dataset = dataset
        self.root = None

    def info_gain(self, column_id, node):
        if self.dataset.quantity_type_list[column_id] == 'D':
            counts = {}
            classes = {}
            for row in node.data_array:
                current = row[column_id]
                currents_class = int(row[-1])
                if current in counts:
                    counts[current] += 1
                    classes[current][currents_class] += 1
                else:
                    counts[current] = 1
                    classes[current] = [0, 0]
                    classes[current][currents_class] += 1

            ordered_key_list = list(counts.keys())
            prob_array = np.array(
                [counts[i] for i in ordered_key_list])/node.data_array.shape[0]
            class_array = np.array([classes[i] for i in ordered_key_list])
            entorpy_array = entropy_arr(class_array)

            # gain = np.sum(prob_array * entorpy_array)
            # infogain = node.entropy - gain
            # return infogain, '-1'
            return node.entropy - np.sum(prob_array * entorpy_array), '-1'

        else:
            median = np.median(node.data_array[:, column_id].astype(int))
            lm = 0
            gm = 0
            lml = [0, 0]
            gml = [0, 0]
            for row in node.data_array:
                current = int(row[column_id])
                currents_class = int(row[-1])
                if current < int(median):
                    lm += 1
                    lml[currents_class] += 1
                else:
                    gm += 1
                    gml[currents_class] += 1

            prob_array = np.array([lm, gm])/node.data_array.shape[0]
            class_array = np.array([lml, gml])
            entorpy_array = entropy_arr(class_array)

            return node.entropy - np.sum(prob_array * entorpy_array), median

    def best_split(self, node):
        max_info_gain = -np.inf
        max_corresponding_id = -1
        max_corresponding_median = -1
        d = {}
        for feature_id in node.remaining_impure_features:
            curr_info_gain, median = self.info_gain(feature_id, node)
            d[self.dataset.column_title[feature_id]] = curr_info_gain
            if curr_info_gain > max_info_gain:
                max_info_gain = curr_info_gain
                max_corresponding_id = feature_id
                max_corresponding_median = median
        # print(d)
        return max_corresponding_id, max_corresponding_median

    def branch(self, node, depth):
        # str_depth = str(depth)
        # spacing = ''
        # if depth <= 9:
        #     str_depth += '   '
        #     spacing += '    '
        # elif depth > 9:
        #     str_depth += '  '
        #     spacing += '    '
        # elif depth > 99:
        #     str_depth += ' '
        #     spacing += '    '
        # str_depth += ('  '*depth)
        # spacing += ('  '*depth)
        # print(str_depth + node.name)

        node.set_entropy()
        node.pred = np.sum(
            node.data_array[:, -1].astype(int))/node.data_array.shape[0]
        if node.entropy == 0 or len(node.remaining_impure_features) == 0:
            node.type_n = 'N'
            # print(spacing + "closed " + str(node.pred))
            # if node.pred == 0.0 or node.pred == 1.0:
            #     print('meh')
            # else:
            #     print('see')
            node.clear_train_garbage()
            return

        cid, median = self.best_split(node)
        node.column_id = cid
        # print(spacing + "testing " +
        #       self.dataset.column_title[cid] + ' ' + self.dataset.quantity_type_list[cid])

        node_type = self.dataset.quantity_type_list[cid]
        if node_type == 'C':
            lt = []
            ge = []
            for row in node.data_array:
                current = int(row[cid])
                if current < median:
                    lt.append(row)
                else:
                    ge.append(row)

            if len(lt) == 0 or len(ge) == 0:
                # node.type_n = 'N'
                # node.pred = np.sum(
                #     node.data_array[:, -1].astype(int))/node.data_array.shape[0]
                node.remaining_impure_features.remove(cid)
                self.branch(node, depth)
                # dont clean here
                # node.clear_train_garbage()
                return

            node.type_n = 'C'
            d_lt = node.remaining_impure_features.copy()
            n_lt = Node(
                'U', self.dataset.column_title[cid]+'_lt_'+str(median), None, np.array(lt), d_lt)
            d_ge = node.remaining_impure_features.copy()
            n_ge = Node(
                'U', self.dataset.column_title[cid]+'_ge_'+str(median), None, np.array(ge), d_ge)
            self.branch(n_lt, depth+1)
            self.branch(n_ge, depth+1)
            node.children = [n_lt, n_ge]
            node.median = median
        else:
            node.type_n = 'D'
            dataset_splits = {}
            for row in node.data_array:
                current = row[cid]
                if current in dataset_splits:
                    dataset_splits[current].append(row)
                else:
                    dataset_splits[current] = [row]

            node.children = []
            node.edge_label = []
            for key in dataset_splits:
                d = node.remaining_impure_features.copy()
                d.remove(cid)
                # key_rev = key
                # if self.dataset.max_variation[cid] > 0:
                #     key_rev = self.dataset.rev_encoding[cid](key_rev)
                n = Node('U', self.dataset.column_title[cid] + '=' + str(key),
                         None, np.array(dataset_splits[key]), d)
                self.branch(n, depth+1)
                node.children.append(n)
                node.edge_label.append(key)

        node.clear_train_garbage()
        return

    def train(self):
        self.root = Node('U', 'root', None, self.dataset.data_array, set(
            [i for i in range(self.dataset.num_features)]))
        self.branch(self.root, 0)

    def traverse(self, y, node):
        if node.type_n == 'N':
            return np.round(node.pred)
        elif node.type_n == 'D':
            to_search = y[node.column_id]
            try:
                id = node.edge_label.index(to_search)
            except ValueError:
                return np.round(node.pred)
            return self.traverse(y, node.children[id])
        elif node.type_n == 'C':
            if int(y[node.column_id]) < node.median:
                return self.traverse(y, node.children[0])
            else:
                return self.traverse(y, node.children[1])

    def predict(self, dataset, pred_filename=None):
        predictions = np.zeros((dataset.data_array.shape[0], 1))
        i = 0
        for row in dataset.data_array:
            predictions[i] = self.traverse(row, self.root)
            i += 1

        if pred_filename != None:
            np.savetxt(pred_filename, predictions, fmt='%i')
        return predictions


if __name__ == '__main__':
    val_type_list = ['i4', 'S20', 'i4', 'S20', 'i4', 'S20',
                     'S20', 'S20', 'S20', 'S20', 'i4', 'i4', 'i4', 'S20', 'i4']
    quant_type_list = ['C', 'D', 'C', 'D', 'C',
                       'D', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'D']
    train_data = Dataset(sys.argv[1], val_type_list, quant_type_list)
    dt = Decision_Tree(train_data)
    dt.train()
    validation_data = Dataset(sys.argv[2], val_type_list, quant_type_list)
    test_data = Dataset(sys.argv[3], val_type_list, quant_type_list)
    dt.predict(test_data, sys.argv[5])
    dt.predict(validation_data, sys.argv[4])
