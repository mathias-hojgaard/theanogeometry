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
    y_chart = M.sym_chart()
    v = M.sym_vector()

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp
    def loss(x,v,y):
        (x1,chart1) = f(x,v)
        y_chart1 = M.update_coords(y,chart1)
        return 1./M.dim.eval()*T.sum(T.sqr(x1 - y_chart1[0]))        
    dloss = lambda x,v,y: T.grad(loss(x,v,y),v)
    dlossf = M.coords_function(lambda x,v,y,y_chart: (loss(x,v,(y,y_chart)),dloss(x,v,(y,y_chart))),v,y,y_chart)

    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
    def shoot(x,y,v0=None):        
        def f(w):
            (z,dz) = dlossf(x,w.astype(theano.config.floatX),y[0],y[1])
            return (tensor(z),tensor(dz))

        if v0 is None:
            v0 = np.zeros(M.dim.eval())
        res = minimize(f, tensor(v0), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': 100})

        return (tensor(res.x),res.fun)

    M.Logf = shoot
