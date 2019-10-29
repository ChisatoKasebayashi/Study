import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
import numpy as np
import cupy


class MyCVAE(chainer.Chain):
    """Conditional Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, n_label):
        super(MyCVAE, self).__init__()
        with self.init_scope():
            self.n_label = n_label
            # encoder
            self.le1_y = L.Linear(n_label, n_h)
            self.le2_y = L.Linear(n_h, n_h)
            self.le0 = L.Linear(n_in, n_h)
            self.le1 = L.Linear(n_h, n_h)
            self.le2_mu = L.Linear(n_h * 2, n_latent)
            self.le2_ln_var = L.Linear(n_h * 2, n_latent)
            # decoder
            self.ld1_y = L.Linear(n_label, n_h)
            self.ld2_y = L.Linear(n_h, n_h)
            self.ld0 = L.Linear(n_latent, n_h)
            self.ld1 = L.Linear(n_h, n_h)
            self.ld2 = L.Linear(n_h * 2, n_in)

    def __call__(self, x, y, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x, y)[0], y, sigmoid)

    def encode(self, x, y):
        h0 = F.dropout(F.relu(self.le0(x)), ratio=0.1)
        h1 = F.dropout(F.tanh(self.le1(h0)), ratio=0.1)
        
        y1 = F.dropout(F.relu(self.le1_y(y)), ratio=0.1)
        h2 = F.dropout(F.tanh(self.le2_y(y1)), ratio=0.1)
        mu = self.le2_mu(F.concat([h1, h2]))
        ln_var = self.le2_ln_var(F.concat([h1, h2]))  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, y, sigmoid=True):
        h0 = F.dropout(F.relu(self.ld0(z)), ratio=0.1)
        h1 = F.dropout(F.tanh(self.ld1(h0)), ratio=0.1)
        
        y1 = F.dropout(F.relu(self.ld1_y(y)), ratio=0.1)
        h2 = F.dropout(F.tanh(self.ld2_y(y1)), ratio=0.1)
        h3 = self.ld2(F.concat([h1, h2]))
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(*args, **kwargs):
            t0 = args[-1]
            xp = cupy.get_array_module(t0)
            t = xp.expand_dims(t0.astype(xp.float32), axis=1)
            #t = xp.eye(self.n_label, dtype=np.float32)[t0]
            x = args[0]

            mu, ln_var = self.encode(x, t)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, t, sigmoid=False)) \
                            / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                        C * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss

        return lf

class MyCVAE_bouble_linear(chainer.Chain):
    """Conditional Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, n_label):
        super(MyCVAE, self).__init__()
        with self.init_scope():
            self.n_label = n_label
            # encoder
            self.le1_y = L.Linear(n_label, n_h)
            self.le1_2 = L.Linear(n_h, n_h)
            self.le2_y = L.Linear(n_h, n_h)
            self.le0 = L.Linear(n_in, n_h)
            self.le0_1 = L.Linear(n_h, n_h)
            self.le1 = L.Linear(n_h, n_h)
            self.le2_mu = L.Linear(n_h * 2, n_latent)
            self.le2_ln_var = L.Linear(n_h * 2, n_latent)
            # decoder
            self.ld1_y = L.Linear(n_label, n_h)
            self.ld1_2 = L.Linear(n_h, n_h)
            self.ld2_y = L.Linear(n_h, n_h)
            self.ld0 = L.Linear(n_latent, n_h)
            self.ld0_1 = L.Linear(n_h, n_h)
            self.ld1 = L.Linear(n_h, n_h)
            self.ld2 = L.Linear(n_h * 2, n_in)

    def __call__(self, x, y, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x, y)[0], y, sigmoid)

    def encode(self, x, y):
        h0 = F.dropout(F.relu(self.le0(x)), ratio=0.1)
        h0_1 = F.dropout(F.relu(self.le0_1(h0)), ratio=0.1)
        h1 = F.dropout(F.tanh(self.le1(h0_1)), ratio=0.1)
        
        y1 = F.dropout(F.relu(self.le1_y(y)), ratio=0.1)
        y1_2 = F.dropout(F.relu(self.le1_2(y1)), ratio=0.1)
        h2 = F.dropout(F.tanh(self.le2_y(y1_2)), ratio=0.1)
        mu = self.le2_mu(F.concat([h1, h2]))
        ln_var = self.le2_ln_var(F.concat([h1, h2]))  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, y, sigmoid=True):
        h0 = F.dropout(F.relu(self.ld0(z)), ratio=0.1)
        h0_1 = F.dropout(F.relu(self.ld0_1(h0)), ratio=0.1)
        h1 = F.dropout(F.tanh(self.ld1(h0_1)), ratio=0.1)
        
        y1 = F.dropout(F.relu(self.ld1_y(y)), ratio=0.1)
        y1_2 = F.dropout(F.relu(self.ld1_2(y1)), ratio=0.1)
        h2 = F.dropout(F.tanh(self.ld2_y(y1_2)), ratio=0.1)
        h3 = self.ld2(F.concat([h1, h2]))
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(*args, **kwargs):
            t0 = args[-1]
            xp = cupy.get_array_module(t0)
            t = xp.expand_dims(t0.astype(xp.float32), axis=1)
            #t = xp.eye(self.n_label, dtype=np.float32)[t0]
            x = args[0]

            mu, ln_var = self.encode(x, t)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, t, sigmoid=False)) \
                            / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                        C * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss

        return lf


class conv_VAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""
    
    def __init__(self, n_in, n_latent, n_h):
        self.n_h = n_h
        super(conv_VAE, self).__init__()
        with self.init_scope():
            '''
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld0 = L.Linear(n_latent, n_h)
            self.ld1 = L.Linear(n_h, n_h)
            self.ld2 = L.Linear(n_h, n_in)
            '''
            w = chainer.initializers.Normal()
            # encoder of convolution
            self.ce1 = L.Convolution2D(n_in, n_h, 4, 2, 3, initialW=w)
            self.ce2 = L.Convolution2D(n_h, n_h*2, 4, 2, 1, initialW=w)
            self.ce3 = L.Convolution2D(n_h*2, n_h*4, 4, 2, 1, initialW=w)
            self.ce4 = L.Convolution2D(n_h*4, n_h*8, 4, 2, 1, initialW=w)
            self.ce4_mu = L.Linear(n_h*8*4, n_latent)
            self.ce4_ln_var = L.Linear(n_h*8*4, n_latent)
            
            #decoder of convolution
            self.ld0 = L.Linear(n_latent, n_h*4*8)
            self.cd1 = L.Deconvolution2D(n_h*8, n_h*4, 4, 2, 1, initialW=w)
            self.cd2 = L.Deconvolution2D(n_h*4, n_h*2, 4, 2, 1, initialW=w)
            self.cd3 = L.Deconvolution2D(n_h*2, n_h, 4, 2, 1, initialW=w)
            self.cd4 = L.Deconvolution2D(n_h, 1, 4, 2, 3, initialW=w)
            
    def forward(self, x, sigmoid=True):
        #"""AutoEncoder"""
        #return self.decode(self.encode(x)[0], sigmoid)
        return self.lf(x)

    def encode(self, x):
        #print(x.shape)
        h1 = F.relu(self.ce1(x))
        #print(h1.shape, 'h1 shape')
        h2 = F.relu(self.ce2(h1))
        #print(h2.shape, 'h2 shape')
        h3 = F.relu(self.ce3(h2))
        #print(h3.shape, 'h3 shape')
        h4 = F.tanh(self.ce4(h3))
        #print(h4.shape, 'h4 shape')
        h4 = F.reshape(h4, (-1, self.n_h*8*4))
        #print(h4.shape, 'h4 shape_')
        mu = self.ce4_mu(h4)
        ln_var = self.ce4_ln_var(h4)
        #print(mu, ln_var)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld0(z))
        #print(h1.shape, '-h1 shape_')
        h1 = F.reshape(h1, (-1, self.n_h*8, 2,2))
        #print(h1.shape, '-h1 shape')
        h2 = F.relu(self.cd1(h1))
        #print(h2.shape, '-h2 shape')
        h3 = F.relu(self.cd2(h2))
        #print(h3.shape, '-h3 shape')
        h4 = F.relu(self.cd3(h3))
        #print(h4.shape, '-h4 shape')
        h5 = F.tanh(self.cd4(h4))
        #print(h5.shape, '-h5 shape')
        if sigmoid:
            return F.sigmoid(h5)
        else:
            return h5
        
    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                #print(z)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

    def lf(self, x):
        k = 1
        beta = 1.0
        mu, ln_var = self.encode(x)
        batchsize = len(mu.data)
        # reconstruction loss
        rec_loss = 0
        for l in six.moves.range(k):
            z = F.gaussian(mu, ln_var)
            rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                / (k * batchsize)
        self.rec_loss = rec_loss
        self.loss = self.rec_loss + \
            beta * gaussian_kl_divergence(mu, ln_var) / batchsize
        chainer.report(
            {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
        return self.loss
    
class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld0 = L.Linear(n_latent, n_h)
            self.ld1 = L.Linear(n_h, n_h)
            self.ld2 = L.Linear(n_h, n_in)

    def forward(self, x, sigmoid=True):
        #"""AutoEncoder"""
        #return self.decode(self.encode(x)[0], sigmoid)
        return self.lf(x)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(self.le2(h1))
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        print(mu, ln_var)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(self.ld0(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

    def lf(self, x):
        k = 1
        beta = 1.0
        mu, ln_var = self.encode(x)
        batchsize = len(mu.data)
        # reconstruction loss
        rec_loss = 0
        for l in six.moves.range(k):
            z = F.gaussian(mu, ln_var)
            rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                / (k * batchsize)
        self.rec_loss = rec_loss
        self.loss = self.rec_loss + \
            beta * gaussian_kl_divergence(mu, ln_var) / batchsize
        chainer.report(
            {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
        return self.loss
