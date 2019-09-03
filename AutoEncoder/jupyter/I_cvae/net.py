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
