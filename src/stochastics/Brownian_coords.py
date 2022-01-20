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
    """ Brownian motion in coordinates """

    x = M.sym_element()
    dW = M.sym_element()
    t = T.scalar()

    def sde_Brownian_coords(dW,t,x,chart):
        gsharpx = M.gsharp((x,chart))
        X = theano.tensor.slinalg.Cholesky()(gsharpx)
        det = -.5*T.tensordot(gsharpx,M.Gamma_g((x,chart)),((0,1),(1,2)))
        sto = T.tensordot(X,dW,(1,0))
        return (det,sto,X)
    
    def chart_update_Brownian_coords(t,x,chart):
        if M.do_chart_update is None:
            return (t,x,chart)

        new_chart = M.centered_chart(M.F((x,chart)))
        new_x = M.update_coords((x,chart),new_chart)[0]

        return theano.ifelse.ifelse(M.do_chart_update((x,chart)),
                (t,x,chart),
                (t,new_x,new_chart)
            )
    
    M.sde_Brownian_coords = sde_Brownian_coords
    M.chart_update_Brownian_coords = chart_update_Brownian_coords
    M.Brownian_coords = lambda x,dWt: integrate_sde(sde_Brownian_coords,integrator_ito,chart_update_Brownian_coords,x[0],x[1],dWt)
    M.Brownian_coordsf = M.coords_function(M.Brownian_coords,dWt)
