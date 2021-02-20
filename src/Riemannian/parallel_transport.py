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

def initialize(M):
    """ Riemannian parallel transport """

    def ode_parallel_transport(x,chart,dx,t,xv,prevchart):
        prevx = xv[0]
        v = xv[1]
        dx = theano.ifelse.ifelse(T.le(T.sum(T.square(chart-prevchart)),1e-5),
                dx,
                M.update_vector((x,chart),prevx,prevchart,dx)
            )
        dv = - T.tensordot(T.tensordot(dx, M.Gamma_g((x,chart)),axes = (0,1)),
                            v, axes = (1,0))
        return T.stack((T.zeros_like(x),dv))
    
    def chart_update_parallel_transport(t,xv,prevchart,x,chart,dx):
        if M.do_chart_update is None:
            return (t,xv,chart)

        prevx = xv[0]
        v = xv[1]
        return (t,theano.ifelse.ifelse(T.le(T.sum(T.square(chart-prevchart)),1e-5),
                                       T.stack((x,v)),
                                       T.stack((x,M.update_vector((prevx,prevchart),x,chart,v)))),
                chart)

    parallel_transport = lambda v,xs,charts,dxs: integrate(ode_parallel_transport,chart_update_parallel_transport,T.stack((xs[0],v)),charts[0],xs,charts,dxs)
    M.parallel_transport = lambda v,xs,charts,dxs: parallel_transport(v,xs,charts,dxs)[1][:,1]
    v = M.sym_vector()
    xs = M.sym_elements()
    charts = M.sym_charts()
    dxs = M.sym_vectors()
    M.parallel_transportf = theano.function([v,xs,charts,dxs],M.parallel_transport(v,xs,charts,dxs))
