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
# Langevin equations https://arxiv.org/abs/1605.09276
###############################################################
def initialize(M):
    q = M.sym_coords()
    p = M.sym_coordscovector()
    
    l = T.scalar()
    s = T.scalar()

    dW = M.sym_element()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q[0])

    def sde_Langevin(dW,t,x,chart,l,s):
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])-l*dq((x[0],chart),x[1])

        X = T.stack((T.zeros((M.dim,M.dim)),s*T.eye(M.dim)))
        det = T.stack((dqt,dpt))
        sto = T.tensordot(X,dW,(1,0))
        return (det,sto,X,T.zeros_like(l),T.zeros_like(s))

    def chart_update_Langevin(t,xp,chart,l,s):
        if M.do_chart_update is None:
            return (t,xp,chart,l,s)

        p = xp[1]
        x = (xp[0],chart)

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_p = M.update_covector(x,new_x,new_chart,p)
        
        return theano.ifelse.ifelse(M.do_chart_update(x),
                (t,xp,chart,l,s),
                (t,T.stack((new_x,new_p)),new_chart,l,s)
            )

    M.sde_Langevin = sde_Langevin
    M.chart_update_Langevin = chart_update_Langevin
    M.Langevin_qp = lambda q,p,l,s,dWt: integrate_sde(sde_Langevin,integrator_ito,chart_update_Langevin,T.stack((q[0],p)),q[1],dWt,l,s)
    M.Langevin_qpf = M.coords_function(M.Langevin_qp,p,l,s,dWt)

    M.Langevin = lambda q,p,l,s,dWt: M.Langevin_qp(q,p,l,s,dWt)[0:3]
    M.Langevinf = M.coords_function(M.Langevin,p,l,s,dWt)
