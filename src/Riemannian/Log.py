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

def initialize(M,f=None):
    """ numerical Riemannian Logarithm map """

    y = M.sym_element()
    v = M.sym_vector()

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp
    loss = lambda x,v,y: 1./M.dim.eval()*T.sum(T.sqr(f(x,v)[0]-y))
    dloss = lambda x,v,y: T.grad(loss(x,v,y),v)
    dlossf = M.coords_function(lambda x,v,y: (loss(x,v,y),dloss(x,v,y)),v,y)

    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
    def shoot(x,y,v0=None):        
        y = M.update_coordsf(y,x[1])
        def f(w):
            (z,dz) = dlossf(x,w.astype(theano.config.floatX),y[0])
            return (tensor(z),tensor(dz))

        if v0 is None:
            v0 = np.zeros(M.dim.eval())
        res = minimize(f, tensor(v0), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': 500})

        return (tensor(res.x),res.fun)

    M.Logf = shoot
