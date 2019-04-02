
# coding: utf-8

# In[4]:


import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


# In[5]:


class CVAE(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h, n_label):
        super(CVAE, self).__init__()
        with self.init_scope():
            #encoder
            self.merge_layer_e_y = L.Linear(n_label, n_h)
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h*2, n_latent)
            self.le2_ln_var = L.Linear(n_h*2, n_latent)
            #dencoder
            self.merge_layer_d_y = L.Linear(n_label, n_h)
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h*2, n_in)
            
    def __call__(self, x, y, sigmoid=True):
        return self.decode(self.encode(x, y)[0], y, sigmoid)
    
    def encode(self, x, y):
        h1 = F.tanh(self.le1(x))
        #print(h1.shape)
        #print(self.merge_layer_e_y(y).shape)
        h2 = F.tanh(self.merge_layer_e_y(y))
        #print(F.concat([h1, self.merge_layer_e_y(y)]).shape)
        mu = self.le2_mu(F.concat([h1, h2]))
        ln_var = self.le2_ln_var(F.concat([h1, h2]))  # log(sigma**2)
        return mu, ln_var
    
    def decode(self, z, y, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = F.tanh(self.merge_layer_d_y(y))
        #print(h1.shape)
        #print('--------')
        #print(y.shape)
        h3 = self.ld2(F.concat([h1,h2 ]))
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3
    def get_loss_func(self, C=1.0, k=1):
        def lf(x, y):
            mu, ln_var = self.encode(x, y)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, y, sigmoid=False))                     / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss +                 C * gaussian_kl_divergence(mu, ln_var) / batchsize
            return self.loss
        return lf

