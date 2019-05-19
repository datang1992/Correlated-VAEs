import numpy as np
import tensorflow as tf
from scipy.special import expit
from sklearn.decomposition import PCA

flags = tf.app.flags
flags.DEFINE_string('output_data_file_name', 'processed_data2.npz', 'Output data file name')
flags.DEFINE_integer('N', 10000, 'Number of vertices')
flags.DEFINE_integer('M', 1000, 'Dimensionality of the data')
flags.DEFINE_integer('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_integer('hidden_size', 300, 'Hidden size for neural networks')
flags.DEFINE_float('tau', 0.99, 'tau value for the prior')

FLAGS = flags.FLAGS

def my_relu(x):
    return np.float64(x >= 0) * x

z = np.zeros((FLAGS.N, FLAGS.latent_dim))
z[0] = np.random.randn(FLAGS.latent_dim)
parent = [-1]
trust = np.zeros((FLAGS.N - 1, 2), dtype=np.int64)

for i in range(1, FLAGS.N):
    parent.append(np.random.randint(i))
    z[i] = FLAGS.tau * z[parent[i]] + np.sqrt(1 - FLAGS.tau ** 2) * np.random.randn(FLAGS.latent_dim)
    trust[i - 1] = np.array([parent[i], i])
    np.random.shuffle(trust[i - 1])

W1 = np.random.randn(FLAGS.latent_dim, FLAGS.hidden_size)
b1 = np.random.randn(FLAGS.hidden_size)
net = my_relu(np.dot(z, W1) + b1)
W2 = np.random.randn(FLAGS.hidden_size, FLAGS.M)
b2 = np.random.randn(FLAGS.M)
ratings = np.random.binomial(1, expit(np.dot(net, W2) + b2))

pca = PCA(n_components=1)
v = pca.fit_transform(z)[:, 0]
c = (v >= np.median(v))

idx = np.random.permutation(FLAGS.N)
idx_reverse = np.arange(FLAGS.N)
idx_reverse[idx] = np.arange(FLAGS.N)
trust = idx_reverse[trust]
np.random.shuffle(trust)
ratings = ratings[idx]
c = c[idx]

f = file(FLAGS.output_data_file_name, 'wb')
np.savez(f, ratings=ratings, trust=trust, c=c, N=FLAGS.N, M=FLAGS.M)
f.close()
