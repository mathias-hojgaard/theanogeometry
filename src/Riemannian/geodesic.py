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

def initialize(M,do_chart_update=None):
    x = M.sym_element()
    v = M.sym_covector()

    def ode_geodesic(t,x,chart):
        dx2t = - T.tensordot(T.tensordot(x[1],
                                         M.Gamma_g((x[0],chart)), axes = [0,1]),
                             x[1],axes = [1,0])
        dx1t = x[1]
        return T.stack((dx1t,dx2t))

    def chart_update_geodesic(t,xv,chart):
        if do_chart_update is None:
            return (t,xv,chart)

        v = xv[1]
        x = (xv[0],chart)

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_v = M.update_vector(x,new_x,new_chart,v)
        
        return theano.ifelse.ifelse(do_chart_update(x),
                (t,xv,chart),
                (t,T.stack((new_x,new_v)),new_chart)
            )

    M.geodesic = lambda x,v: integrate(ode_geodesic,chart_update_geodesic,T.stack((x[0],v)), x[1])
    M.geodesicf = M.coords_function(M.geodesic,v)

    def Exp(x,v):
        curve = M.geodesic(x,v)
        x = curve[1][-1,0]
        chart = curve[2][-1]
        return(x,chart)
    M.Exp = Exp
    M.Expf = M.coords_function(M.Exp,v)
    def Expt(x,v):
        curve = M.geodesic(x,v)
        xs = curve[1][:,0]
        charts = curve[2]
        return(xs,charts)
    M.Expt = Expt
    M.Exptf = M.coords_function(M.Expt,v)

