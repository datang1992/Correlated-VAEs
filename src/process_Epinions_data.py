import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('rating_count_threshold', 100, 'Rating count threshold for products')
flags.DEFINE_string('input_data_file_name', 'original_data.npz', 'Input data file name')
flags.DEFINE_string('output_data_file_name', 'processed_data.npz', 'Output data file name')
flags.DEFINE_float('train_ratio', 0.9, 'Ratio for the training set for the ratings data')
flags.DEFINE_integer('edge_subset_size', 10000, 'Subset size for edges.')

FLAGS = flags.FLAGS

def get_edge_weight(N, edge):
    e = [[] for i in range(N)]
    for j in range(edge.shape[0]):
        e[edge[j, 0]].append(edge[j, 1])
        e[edge[j, 1]].append(edge[j, 0])
    visited = [False for i in range(N)]
    e_s = set((edge[:, 0] * N + edge[:, 1]).tolist())
    w_m = {}
    for i in range(N):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            cc = []
            while len(queue) > 0:
                now = queue[0]
                cc.append(now)
                for j in e[now]:
                    if not visited[j]:
                        visited[j] = True
                        queue.append(j)
                queue.pop(0)
            N_cc = len(cc)
            v2n = {cc[j]: j for j in range(N_cc)}
            L = np.zeros((N_cc, N_cc))
            for u in cc:
                for v in e[u]:
                    L[v2n[u], v2n[u]] += 1
                    L[v2n[u], v2n[v]] -= 1
            L_pinv = np.linalg.pinv(L)
            for u in cc:
                for v in e[u]:
                    num = u * N + v
                    if num  in e_s:
                        w = L_pinv[v2n[u], v2n[u]] + L_pinv[v2n[v], v2n[v]]\
                                - L_pinv[v2n[u], v2n[v]] - L_pinv[v2n[v], v2n[u]]
                        w_m[num] = w

    e_w = np.zeros(edge.shape[0])
    for i in range(edge.shape[0]):
        e_w[i] = w_m[edge[i, 0] * N + edge[i, 1]]
    return e_w

f = file(FLAGS.input_data_file_name, 'rb')
a = np.load(f)
ratings_data = a['ratings_data']
trust_data = a['trust_data']
trust_data = trust_data[:, :2]
f.close()

N = max(np.max(trust_data), np.max(ratings_data[:, 0]))
M = np.max(ratings_data[:, 1])

idx = np.where(trust_data[:, 0] != trust_data[:, 1])[0]
trust_data = trust_data[idx]
trust_data -= 1
ratings_data[:, :2] -= 1

counts = np.zeros(M, dtype=np.int64)
for i in range(ratings_data.shape[0]):
    counts[ratings_data[i, 1]] += 1
product_id = np.where(counts >= FLAGS.rating_count_threshold)[0]
id_map3 = -np.ones(M, dtype=np.int64)
id_map3[product_id] = np.arange(product_id.shape[0])
ratings_data[:, 1] = id_map3[ratings_data[:, 1]]
ratings_data = ratings_data[np.where(ratings_data[:, 1] > -1)[0]]
M = product_id.shape[0]

s1 = set((trust_data[:, 0] * N + trust_data[:, 1]).tolist())
s2 = set((trust_data[:, 1] * N + trust_data[:, 0]).tolist())
edge_id = np.asarray(list(s1.intersection(s2)))
trust_data = np.asarray([edge_id / N, edge_id % N]).transpose()

s3 = set(ratings_data[:, 0].tolist())
s4 = set(trust_data[:, 0].tolist())
s5 = set(trust_data[:, 1].tolist())
new_l = np.asarray(list(s3.intersection(s4.intersection(s5))))
id_map = -np.ones(N, dtype=np.int64)
id_map[new_l] = np.arange(new_l.shape[0])
trust_data = id_map[trust_data]
trust_data = trust_data[np.where(np.min(trust_data, 1) > -1)[0]]
ratings_data[:, 0] = id_map[np.copy(ratings_data[:, 0])]
ratings_data = ratings_data[np.where(ratings_data[:, 0] > -1)[0]]
N = new_l.shape[0]

s6 = set(ratings_data[:, 0].tolist())
s7 = set(trust_data[:, 0].tolist())
s8 = set(trust_data[:, 1].tolist())
new_l2 = np.asarray(list(s6.intersection(s7.intersection(s8))))
id_map2 = -np.ones(N, dtype=np.int64)
id_map2[new_l2] = np.arange(new_l2.shape[0])
trust_data = id_map2[trust_data]
trust_data = trust_data[np.where(np.min(trust_data, 1) > -1)[0]]
ratings_data[:, 0] = id_map2[np.copy(ratings_data[:, 0])]
ratings_data = ratings_data[np.where(ratings_data[:, 0] > -1)[0]]
N = new_l2.shape[0]

ratings = csr_matrix((np.ones(ratings_data.shape[0]), (ratings_data[:, 0], ratings_data[:, 1])), (N, M)).toarray()
trust = trust_data[np.where(trust_data[:, 0] < trust_data[:, 1])[0]]
for i in range(N):
    np.random.shuffle(trust[i])

rand_id_map1 = np.random.permutation(M)
ratings = ratings[:, rand_id_map1]

rand_id_map2 = np.random.permutation(N)
ratings = ratings[rand_id_map2]
rand_id_map2_inverse = -np.ones(N, dtype=np.int64)
rand_id_map2_inverse[rand_id_map2] = np.arange(N)
trust = rand_id_map2_inverse[trust]

edge = [[] for i in range(N)]
edge_train = [[] for i in range(N)]
edge_test = [[] for i in range(N)]
for i in range(trust.shape[0]):
    edge[trust[i, 0]].append(trust[i, 1])
    edge[trust[i, 1]].append(trust[i, 0])

trust_test = []
for i in range(N):
    n = len(edge[i])
    n_test = int(np.ceil(n * ((1 - FLAGS.train_ratio) / 2)))
    ind = np.random.permutation(n)
    for j in range(n):
        if j < n_test:
            if edge[i][ind[j]] not in edge_test[i]:
                edge_test[i].append(edge[i][ind[j]])
                edge_test[edge[i][ind[j]]].append(i)
                trust_test.append([i, edge[i][ind[j]]])

trust_train = []
for i in range(N):
    for j in edge[i]:
        if j not in edge_test[i]:
            edge_train[i].append(j)
            if i < j:
                trust_train.append([i, j])

trust_train = np.asarray(trust_train)
trust_test = np.asarray(trust_test)
np.random.shuffle(trust_train)
np.random.shuffle(trust_test)
for i in range(trust_train.shape[0]):
    np.random.shuffle(trust_train[i])
for i in range(trust_test.shape[0]):
    np.random.shuffle(trust_test[i])

edge_id_subset = np.random.permutation(trust_train.shape[0])[:FLAGS.edge_subset_size]

edge_weight = get_edge_weight(N, trust_train)

f = file(FLAGS.output_data_file_name, 'wb')
np.savez(f, ratings=ratings, trust_train=trust_train, trust_test=trust_test, edge_id_subset=edge_id_subset, edge_weight=edge_weight, N=N, M=M)
f.close()
