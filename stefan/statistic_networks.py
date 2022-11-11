import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Conv2D, Flatten, GaussianDropout
from tensorflow.keras.models import Sequential


class MAB(tf.keras.Model):
    def __init__(self):
        super(MAB, self).__init__()
        
        self.att = MultiHeadAttention(key_dim=128, num_heads=4, dropout=0.3)
        self.fc = Sequential()
        self.fc.add(Dense(256, activation='relu'))
        self.fc.add(Dense(128))
        self.ln_pre = LayerNormalization()
        self.ln_post = LayerNormalization()
        
    def call(self, x, y, **kwargs):
        h = x + self.att(x, y, y, **kwargs)
        if self.ln_pre is not None:
            h = self.ln_pre(h, **kwargs)
        out = h + self.fc(h)
        if self.ln_post is not None:
            out = self.ln_post(out, **kwargs)
        return out
    

class PMA(tf.keras.Model):
    def __init__(self):
        super(PMA, self).__init__()
        self.mab = MAB()
        init = tf.keras.initializers.GlorotUniform()
        self.seed_vec = tf.Variable(init(shape=(4, 128)), name='seed_vec', trainable=True)
        
    def call(self, x, **kwargs):
        batch_size = x.shape[0]
        z = self.mab(tf.stack([self.seed_vec] * x.shape[0]), x, **kwargs)
        return z


class ResidualFC(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ResidualFC, self).__init__(**kwargs)
        self.pre = Sequential([
            Dense(256, activation='relu'),
            LayerNormalization()
        ])
        self.res_net = Sequential([
            Dense(256, activation='relu'),
            LayerNormalization(),
            Dense(256, activation='relu'),
            LayerNormalization(),
            Dense(256)
        ])
        self.post = Sequential([
            Dense(256, activation='relu'),
            LayerNormalization()
        ])
        
    def __call__(self, x, **kwargs):
        pre = self.pre(x, **kwargs)
        res = tf.nn.relu(pre + self.res_net(pre, **kwargs))
        post = self.post(res, **kwargs)
        return post
        

class AttentiveStatisticNetwork(tf.keras.Model):
    def __init__(self, dim=10, **kwargs):
        super(AttentiveStatisticNetwork, self).__init__(**kwargs)
        
        self.pre = ResidualFC()
        self.post = ResidualFC()
        self.out = Dense(dim)
        self.pma = PMA()
    
    def __call__(self, x, **kwargs):
        pre = self.pre(x, **kwargs)
        pooled = self.pma(pre, **kwargs)
        pooled = tf.reshape(pooled, (pooled.shape[0], pooled.shape[1] * pooled.shape[2]))
        out = self.out(self.post(pooled, **kwargs))
        return out