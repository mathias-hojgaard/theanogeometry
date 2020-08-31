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
def initialize(M,k=None,do_chart_update=None):
    q = M.sym_coords()
    p = M.sym_coordscovector()
    
    sigmas_x = T.vector()
    sigmas_a = T.vector()

    dW = M.sym_element()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q[0])
    
    # noise basis
    if k is None: # use landmark kernel per default
        k = M.k
    
    q1 = M.sym_coords()
    q2 = T.vector()
    k_q = lambda q1,q2: k(q1.reshape((-1,M.m)).dimshuffle(0,'x',1)-q2.reshape((-1,M.m)).dimshuffle('x',0,1))
    K = lambda q1,q2: (k_q(q1,q2)[:,:,np.newaxis,np.newaxis]*T.eye(M.m)[np.newaxis,np.newaxis,:,:]).dimshuffle((0,2,1,3)).reshape((M.dim,-1))

    def sde_Eulerian(dW,t,x,chart,sigmas_x,sigmas_a):
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        
        lq = x[0]
        sigmas_adW = (sigmas_a.reshape((-1,M.m))*dW[:,np.newaxis]).flatten()
        sigmadWq = T.tensordot(K(x[0],sigmas_x),sigmas_adW,(1,0))
        sigmadWp = T.tensordot(
             T.jacobian(
                 T.tensordot(
                     K(lq,theano.gradient.disconnected_grad(sigmas_x)),
                     sigmas_adW,(1,0)).flatten(),
                 lq),
            x[1],(1,0))
    
        X = None # to be implemented
        det = T.stack((dqt,dpt))
        sto = T.stack((sigmadWq,sigmadWp))
        return (det,sto,X,sigmas_x,sigmas_a)

    def chart_update_Eulerian(t,xp,chart,sigmas_x,sigmas_a):
        if do_chart_update is None:
            return (t,xp,chart,sigmas_x,sigmas_a)

        p = xp[1]
        x = (xp[0],chart)

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_p = M.update_covector(x,new_x,new_chart,p)
        
        return theano.ifelse.ifelse(do_chart_update(x),
                (t,xp,chart,sigmas_x,sigmas_a),
                (t,T.stack((new_x,new_p)),new_chart,sigmas_x,sigmas_a)
            )

    M.sde_Eulerian = sde_Eulerian
    M.chart_update_Eulerian = chart_update_Eulerian
    M.Eulerian_qp = lambda q,p,sigmas_x,sigmas_a,dWt: integrate_sde(sde_Eulerian,integrator_ito,chart_update_Eulerian,T.stack((q[0],p)),q[1],dWt,sigmas_x,sigmas_a)
    M.Eulerian_qpf = M.coords_function(M.Eulerian_qp,p,sigmas_x,sigmas_a,dWt)

    M.Eulerian = lambda q,p,sigmas_x,sigmas_a,dWt: M.Eulerian_qp(q,p,sigmas_x,sigmas_a,dWt)[0:3]
    M.Eulerianf = M.coords_function(M.Eulerian,p,sigmas_x,sigmas_a,dWt)
