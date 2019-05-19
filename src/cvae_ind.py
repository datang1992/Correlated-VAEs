import itertools
import numpy as np
import os
import tensorflow as tf
import time

distributions = tf.distributions

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'processed_data.npz', 'Directory for data')

flags.DEFINE_integer('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
flags.DEFINE_integer('batch_size2', 256, 'Minibatch size 2')
flags.DEFINE_integer('print_every', 10000, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 300, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 300000, 'number of iterations')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('tau', 0.99, 'tau value for the prior')
flags.DEFINE_float('init_scale', 0.1, 'init scale value.')

FLAGS = flags.FLAGS

def weight_variable(shape, name, init_type='truncated_normal'):
  if init_type == 'truncated_normal':
    init_val = tf.truncated_normal(shape, stddev=FLAGS.init_scale)
  with tf.variable_scope('weights'):
    variable = tf.get_variable(name, initializer=init_val)
  return variable

def get_e(N, trust):
  e =[[] for i in range(N)]
  for j in range(trust.shape[0]):
    e[trust[j, 0]].append(trust[j, 1])
    e[trust[j, 1]].append(trust[j, 0])
  return e

def load_data(data_dir):
  npzfiles = np.load(data_dir)
  N = int(np.round(npzfiles['N']))
  M = int(np.round(npzfiles['M']))
  ratings = npzfiles['ratings']
  trust_train = np.int64(np.round(npzfiles['trust_train']))
  trust_test = np.int64(np.round(npzfiles['trust_test']))
  e_train = get_e(N, trust_train)
  e_test = get_e(N, trust_test)
  edge_id_subset = np.int32(np.round(npzfiles['edge_id_subset']))
  edge_weight = npzfiles['edge_weight']
  return N, M, ratings, trust_train, trust_test, e_train, e_test, edge_id_subset, edge_weight

def inference_network(x, M, latent_dim, hidden_size):
  W1 = weight_variable([M, hidden_size], 'W1_inf')
  b1 = weight_variable([hidden_size], 'b1_inf')
  net = tf.nn.relu(tf.matmul(x, W1) + b1)
  W2 = weight_variable([hidden_size, latent_dim * 2], 'W2_inf')
  b2 = weight_variable([latent_dim * 2], 'b2_inf')
  gaussian_params = tf.matmul(net, W2) + b2
  mu = gaussian_params[:, :latent_dim]
  sigma = tf.nn.softplus(gaussian_params[:, latent_dim:])
  return mu, sigma, W1, b1, W2, b2

def generative_network(z, latent_dim, hidden_size, M):
  W1 = weight_variable([latent_dim, hidden_size], 'W1_gen')
  b1 = weight_variable([hidden_size], 'b1_gen')
  net = tf.nn.relu(tf.matmul(z, W1) + b1)
  W2 = weight_variable([hidden_size, M], 'W2_gen')
  b2 = weight_variable([M], 'b2_gen')
  logits = tf.matmul(net, W2) + b2
  return logits, W1, b1, W2, b2

def get_icrr(e_train, e_test):
  N = len(e_test)
  a = np.arange(N + 1).astype(np.float64)
  a[1:] = 1.0 / a[1:]
  for i in range(1, N + 1):
    a[i] += a[i - 1]
  len_train = np.int64([len(e_train[i]) for i in range(N)])
  len_non_train = N - 1 - len_train
  len_test = np.int64([len(e_test[i]) for i in range(N)])
  return a[len_train], a[len_test], np.mean(a[len_non_train] / np.float64(len_non_train) * np.float64(len_test) / a[len_test])

def get_ncrr(e_train, e_test, icrr_train, icrr_test, np_q_mu, np_q_sigma):
  np_q_para = np.hstack((np_q_mu, np_q_sigma))
  N = np_q_para.shape[0]
  np_q_para_square_sum = np.sum(np_q_para ** 2, 1)
  dis = np_q_para_square_sum + np_q_para_square_sum.reshape(-1, 1) -\
          2 * np.dot(np_q_mu, np_q_mu.transpose())
  ncrr_train = 0.
  crr_test = np.zeros(N)
  max_val = np.max(dis) + 1.0
  for i in range(N):
    row = np.copy(dis[i])
    row[i] = max_val
    crr_train = 0.
    for j in e_train[i]:
      crr_train += 1.0 / (np.sum(row <= row[j]))
    row[np.int64(e_train[i])] = max_val
    for j in e_test[i]:
      crr_test[i] += 1.0 / (np.sum(row <= row[j]))
    if icrr_train[i] > 0:
      ncrr_train += crr_train / icrr_train[i]
  ncrr_train /= np.sum(icrr_train > 0)
  return ncrr_train, np.mean(crr_test / icrr_test)

def train():
  N, M, ratings, trust_train, trust_test, e_train, e_test, edge_id_subset, edge_weight = load_data(FLAGS.data_dir)
  icrr_train, icrr_test, rcrr_test = get_icrr(e_train, e_test)
  trust_subset = trust_train[edge_id_subset]
  edge_weight_subset = edge_weight[edge_id_subset]

  N2 = trust_train.shape[0]

  with tf.name_scope('data'):
    x_w = tf.placeholder(tf.float32, [None, M])
    e = tf.placeholder(tf.int32, [None, 2])
    e_w = tf.placeholder(tf.float32, [None])
    T = tf.placeholder(tf.int32, [])
    x = x_w[:T]
    x_e = tf.gather(x_w, e)


  with tf.variable_scope('variational'):
    q_mu_w, q_sigma_w, W1_inf, b1_inf, W2_inf, b2_inf = inference_network(x=x_w,
                                      M=M,
                                      latent_dim=FLAGS.latent_dim,
                                      hidden_size=FLAGS.hidden_size)
    q_mu = q_mu_w[:T]
    q_sigma = q_sigma_w[:T]
    q_z = distributions.Normal(loc=q_mu, scale=q_sigma)

  with tf.variable_scope('model'):
    p_x_given_z_logits, W1_gen, b1_gen, W2_gen, b2_gen = generative_network(z=q_z.sample(),
                                            latent_dim=FLAGS.latent_dim,
                                            hidden_size=FLAGS.hidden_size,
                                            M=M)
    p_x_given_z = distributions.Multinomial(total_count=tf.reduce_sum(x, 1), logits=p_x_given_z_logits)


  kl_s = tf.reduce_mean(tf.reduce_sum(0.5 * ((q_mu ** 2) + (q_sigma ** 2))\
                            - tf.log(q_sigma) - 0.5, 1))


  q_mu2 = tf.gather(q_mu_w, e)
  q_sigma2 = tf.gather(q_sigma_w, e)
  kl_p = tf.reduce_mean(tf.reduce_sum(0.5 * ((q_mu2[:, 0] ** 2) + (q_mu2[:, 1] ** 2)\
            - 2 * q_mu2[:, 0] * q_mu2[:, 1] * FLAGS.tau\
            + (q_sigma2[:, 0] ** 2) + (q_sigma2[:, 1] ** 2))\
            / (1. - FLAGS.tau ** 2)\
            - 0.5 * ((q_mu2[:, 0] ** 2) + (q_sigma2[:, 0] ** 2)\
            + (q_mu2[:, 1] ** 2) + (q_sigma2[:, 1] ** 2))\
            + 0.5 * tf.log(1. - FLAGS.tau ** 2), 1) * e_w)

  kl = kl_s + (kl_p * N2) / N

  expected_log_likelihood = tf.reduce_mean(p_x_given_z.log_prob(x))
  
  elbo = expected_log_likelihood - kl

  nelbo = -elbo

  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  train_op = optimizer.minimize(nelbo)

  init_op = tf.global_variables_initializer()

  sess = tf.InteractiveSession()
  sess.run(init_op)

  print 'Random Test CRR: {0:.5f}'.format(rcrr_test)
  t0 = time.time()
  cur_pos = 0
  cur_pos2 = 0
  best_train_ncrr = 0.
  best_test_ncrr = 0.
  best_train_nelbo = np.inf
  np_e = np.hstack((np.arange(FLAGS.batch_size, FLAGS.batch_size + FLAGS.batch_size2).reshape(-1, 1), np.arange(FLAGS.batch_size + FLAGS.batch_size2, FLAGS.batch_size + FLAGS.batch_size2 * 2).reshape(-1, 1)))
  for i in range(FLAGS.n_iterations):
    if cur_pos + FLAGS.batch_size <= N:
      np_x = ratings[cur_pos:cur_pos + FLAGS.batch_size]
    else:
      np_x = np.vstack((ratings[cur_pos:], ratings[:cur_pos + FLAGS.batch_size - N]))
    cur_pos = (cur_pos + FLAGS.batch_size) % N
    if cur_pos2 + FLAGS.batch_size2 <= N2:
      np_edge = trust_train[cur_pos2:cur_pos2 + FLAGS.batch_size2]
      np_e_w = edge_weight[cur_pos2:cur_pos2 + FLAGS.batch_size2]
    else:
      np_edge = np.vstack((trust_train[cur_pos2:], trust_train[:cur_pos2 + FLAGS.batch_size2 - N2]))
      np_e_w = np.hstack((edge_weight[cur_pos2:], edge_weight[:cur_pos2 + FLAGS.batch_size2 - N2]))
    cur_pos2 = (cur_pos2 + FLAGS.batch_size2) % N2
    np_x2 = ratings[np_edge]
    np_x_w = np.vstack((np_x, np_x2[:, 0], np_x2[:, 1]))
    sess.run(train_op, {x_w: np_x_w, T: FLAGS.batch_size, e: np_e, e_w: np_e_w})
    
    if (i + 1) % FLAGS.print_every == 0:
      np_batch_elbo = sess.run(nelbo, {x_w: np_x_w, T: FLAGS.batch_size, e: np_e, e_w: np_e_w})
      [np_q_mu, np_q_sigma, np_train_elbo] = sess.run([q_mu, q_sigma, nelbo], {x_w: ratings, T: N, e: trust_subset, e_w: edge_weight_subset})
      train_ncrr, test_ncrr = get_ncrr(e_train, e_test, icrr_train, icrr_test, np_q_mu, np_q_sigma)
      if train_ncrr > best_train_ncrr and np_train_elbo < best_train_nelbo:
        best_train_ncrr = train_ncrr
        best_test_ncrr = test_ncrr
        best_train_nelbo = np_train_elbo
      print('Iteration: {0:d} Batch NELBO: {1:.3f} Train NELBO: {2:.3f} Test NCRR: {3:.4f} Best Test NCRR: {4:.4f} Time: {5:.2f}'.format(
          i + 1,
          np_batch_elbo, np_train_elbo,
          test_ncrr,
          best_test_ncrr,
          time.time() - t0))

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
