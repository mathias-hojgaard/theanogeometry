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

###############################################################
# Eulerian / stochastic EPDiff acting on landmarks
###############################################################
def initialize(M,k=None,):
    q = M.coords()
    p = M.coordscovector()
    
    sigmas_x = T.vector()
    sigmas_a = T.vector()

    dW = M.element()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q)
    
    # noise basis
    if k is None: # use landmark kernel per default
        k = M.k
    
    q1 = M.coords()
    q2 = T.vector()
    k_q = lambda q1,q2: k(q1.reshape((-1,M.m)).dimshuffle(0,'x',1)-q2.reshape((-1,M.m)).dimshuffle('x',0,1))
    K = lambda q1,q2: (k_q(q1,q2)[:,:,np.newaxis,np.newaxis]*T.eye(M.m)[np.newaxis,np.newaxis,:,:]).dimshuffle((0,2,1,3)).reshape((M.dim,-1))

    def sde_Eulerian(dW,t,x,sigmas_x,sigmas_a):
        lq = x[0]
        lp = x[1]
        dqt = dq(lq,lp)
        dpt = dp(lq,lp)
        
        sigmas_adW = (sigmas_a.reshape((-1,M.m))*dW[:,np.newaxis]).flatten()
        sigmadWq = T.tensordot(K(x[0],sigmas_x),sigmas_adW,(1,0))
        sigmadWp = T.tensordot(
             T.jacobian(
                 T.tensordot(
                     K(lq,theano.gradient.disconnected_grad(sigmas_x)),
                     sigmas_adW,(1,0)).flatten(),
                 lq),
            lp,(1,0))
    
        X = None # to be implemented
        det = T.stack((dqt,dpt))
        sto = T.stack((sigmadWq,sigmadWp))
        return (det,sto,X,sigmas_x,sigmas_a)
    M.Eulerian_qp = lambda q,p,sigmas_x,sigmas_a,dWt: integrate_sde(sde_Eulerian,integrator_ito,T.stack((q,p)),dWt,sigmas_x,sigmas_a)
    M.Eulerian_qpf = theano.function([q,p,sigmas_x,sigmas_a,dWt], M.Eulerian_qp(q,p,sigmas_x,sigmas_a,dWt))

    M.Eulerian = lambda q,p,sigmas_x,sigmas_a,dWt: M.Eulerian_qp(q,p,sigmas_x,sigmas_a,dWt)[0:2]
    M.Eulerianf = theano.function([q,p,sigmas_x,sigmas_a,dWt], M.Eulerian(q,p,sigmas_x,sigmas_a,dWt))


