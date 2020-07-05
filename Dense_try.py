from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
import tensorflow as tf


class Dense_try(Layer):

    def __init__(self, units,
                
                 activation=None,
                 use_bias=True,
                 use_svd=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense_try, self).__init__(**kwargs)
        self.units = units
      
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.use_svd = use_svd
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
  
        else:
            self.bias = None
            

      

        super(Dense_try, self).build(input_shape)


    def call(self, X):
        
        output = K.dot(X, self.kernel)
        if self.use_bias:
            
            perturbed_bias = self.bias
            output = K.bias_add(output, perturbed_bias)
        if self.use_svd:
            print(self.kernel)
            s, u, v = tf.linalg.svd(self.kernel)
            mean= tf.reduce_mean(s)
            zero = tf.zeros_like(s)
            s=tf.where(s<mean,x=zero,y=s)
            print(s)
            self.kernel = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
            print(self.kernel)


            
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    