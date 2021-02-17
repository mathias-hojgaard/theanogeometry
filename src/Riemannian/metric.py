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
from src.linalg import *

def initialize(M,truncate_high_order_derivatives=False):
    """ add metric related structures to manifold """

    def truncate_derivatives(e):
        if truncate_high_order_derivatives:
            return theano.gradient.disconnected_grad(e)
        else:
            return e

    d = M.dim
    #x = M.element()
    #u = T.matrix()

    if hasattr(M, 'g'):
        if not hasattr(M, 'gsharp'):
            M.gsharp = lambda x: T.nlinalg.matrix_inverse(M.g(x))
    elif hasattr(M, 'gsharp'):
        if not hasattr(M, 'g'):
            M.g = lambda x: T.nlinalg.matrix_inverse(M.gsharp(x))
    else:
        raise ValueError('no metric or cometric defined on manifold')
    M.gf = M.coords_function(M.g)
    M.gsharpf = M.coords_function(M.gsharp)

    M.Dg = lambda x: T.jacobian(M.g(x).flatten(),x[0]).reshape((d,d,d)) # Derivative of metric
    M.Dgf = M.coords_function(M.Dg)

    ##### Measure
    M.mu_Q = lambda x: 1./T.nlinalg.Det()(M.g(x))
    M.mu_Qf = M.coords_function(M.mu_Q)

    ### Determinant
    M.determinant = lambda x,A: T.nlinalg.Det()(T.tensordot(M.g(x),A,(1,0)))
    M.LogAbsDeterminant = lambda x,A: LogAbsDet()(T.tensordot(M.g(x),A,(1,0)))
    A = T.matrix()
    M.Determinantf = M.coords_function(M.determinant,A)
    M.LogAbsDeterminantf = M.coords_function(M.LogAbsDeterminant,A)

    ##### Sharp and flat map:
#    M.Dgsharp = lambda q: T.jacobian(M.gsharp(q).flatten(),q).reshape((d,d,d)) # Derivative of sharp map
    v = M.sym_vector()
    p = M.sym_covector()
    M.flat = lambda x,v: T.dot(M.g(x),v)
    M.flatf = M.coords_function(M.flat,v)
    M.sharp = lambda x,p: T.dot(M.gsharp(x),p)
    M.sharpf = M.coords_function(M.sharp,p)

    ##### Christoffel symbols
    # \Gamma^i_{kl}, indices in that order
    M.Gamma_g = lambda x: 0.5*(T.tensordot(M.gsharp(x),truncate_derivatives(M.Dg(x)),axes = [1,0])
                   +T.tensordot(M.gsharp(x),truncate_derivatives(M.Dg(x)),axes = [1,0]).dimshuffle(0,2,1)
                   -T.tensordot(M.gsharp(x),truncate_derivatives(M.Dg(x)),axes = [1,2]))
    M.Gamma_gf = M.coords_function(M.Gamma_g)

    # Inner Product from g
    w = M.sym_vector()
    M.dot = lambda x,v,w: T.dot(T.dot(M.g(x),w),v)
    M.dotf = M.coords_function(M.dot,v,w)
    M.norm = lambda x,v: T.sqrt(M.dot(x,v,v))
    M.normf = M.coords_function(M.norm,v)
    pp = M.sym_covector()
    M.dotsharp = lambda x,p,pp: T.dot(T.dot(M.gsharp(x),pp),p)
    M.dotsharpf = M.coords_function(M.dotsharp,pp,p)
    M.conorm = lambda x,p: T.sqrt(M.dotsharp(x,p,p))
    M.conormf = M.coords_function(M.conorm,p)

    ##### Gram-Schmidt and basis
    M.gramSchmidt = lambda x,u: (GramSchmidt_f(M.dotf))(x,u)
    M.orthFrame = lambda x: T.slinalg.Cholesky()(M.gsharp(x))
    M.orthFramef = M.coords_function(M.orthFrame)

    ##### Hamiltonian
    p = M.sym_covector()
    M.H = lambda q,p: 0.5*T.dot(p,T.dot(M.gsharp(q),p))
    M.Hf = M.coords_function(M.H,p)

    # gradient, divergence, and Laplace-Beltrami
    M.grad = lambda x,f: M.sharp(x,T.grad(f(x),x))
    M.div = lambda x,X: T.nlinalg.trace(T.jacobian(X(x),x))+.5*T.dot(X(x),T.grad(linalg.LogAbsDet()(M.g(x)),x))
    M.Laplacian = lambda x,f: M.div(x,lambda x: M.grad(x,f))
