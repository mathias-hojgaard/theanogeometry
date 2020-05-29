# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

from src.setup import *
from src.utils import *

def initialize(G):
    """ Lie-Poisson geodesic integration """

    assert(G.invariance == 'left')

    g = G.sym_element() # \RR^{NxN} matrix
    q = G.sym_Vvector() # element in psi coordinates
    v = G.sym_coordsvector() # \RR^G_dim tangent vector in coordinates
    mu = G.sym_Vcovector() # \RR^G_dim LA cotangent vector in coordinates

    def ode_LP(t,mu):
        dmut = G.coad(G.dHminusdmu(mu),mu)
        return dmut
    G.LP = lambda mu: integrate(ode_LP,None,mu,None)
    G.LPf = theano.function([mu], G.LP(mu))

    # reconstruction
    def ode_LPrec(mu,t,g):
        dgt = G.dL(g,G.e,G.VtoLA(G.dHminusdmu(mu)))
        return dgt
    G.LPrec = lambda g,mus: integrate(ode_LPrec,None,g,None,mus)
    mus = T.matrix() # mu for each time step
    G.LPrecf = G.function(G.LPrec,mus)

    ### geodesics
    G.coExpLP = lambda g,mu: G.LPrec(g,G.LP(mu)[1])[1][-1]
    G.ExpLP = lambda g,v: G.coExpLP(g,G.flatV(v))
    G.coExpLPt = lambda g,mu: G.LPrec(g,G.LP(mu)[1])
    G.ExpLPt = lambda g,v: G.coExpLPt(g,G.flatV(v))
    G.DcoExpLP = lambda g,mu: (
        T.jacobian(G.coExp(g,mu).flatten(),g).reshape(G.N,G.N,G.N,G.N),
        T.jacobian(G.coExp(g,mu).flatten(),mu).reshape(G.N,G.N,G.dim)
        )
    G.ExpLPf = G.function(G.ExpLP,v)
    G.ExpLPtf = G.function(G.ExpLPt,v)
    G.coExpLPf = G.function(G.coExpLP,mu)
    G.coExpLPtf = G.function(G.coExpLPt,mu)
    #loss = 1./G_emb_dim*T.sum(T.sqr(Exp(g,mu)-h))
    #dloss = (T.grad(loss,g),T.grad(loss,g))
    #lossf = theano.function([g,mu,h], loss)
    #dlossf = theano.function([g,mu,h], [loss, dloss[0], dloss[1]])
