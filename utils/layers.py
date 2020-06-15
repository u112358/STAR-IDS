import tensorflow.compat.v1 as tf

def autoencoder(input, hidden_layers=None, name = None, reuse=None):
    if hidden_layers==None:
        hidden_layers = [32,16]
    with tf.variable_scope(name_or_scope=name, reuse=reuse):
        current = input
        for i in range(len(hidden_layers)):
            layer_name = 'layer_'+str(i)
            current = tf.layers.dense(current, hidden_layers[i], tf.nn.relu, name=layer_name, kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))
        return current

def regressor(input, output_dim, name='regressor'):
    current = input
    current = tf.layers.dense(current, output_dim, tf.nn.tanh, name=name)
    return current