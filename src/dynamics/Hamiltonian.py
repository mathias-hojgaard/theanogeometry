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
# geodesic integration, Hamiltonian form                      #
###############################################################
def initialize(M,do_chart_update=None):
    q = M.sym_coords()
    p = M.sym_coordscovector()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q[0])

    def ode_Hamiltonian(t,x,chart):
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        return T.stack((dqt,dpt))

    def chart_update_Hamiltonian(t,xp,chart):
        if do_chart_update is None:
            return (t,xp,chart)

        p = xp[1]
        x = (xp[0],chart)

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_p = M.update_covector(x,new_x,new_chart,p)
        
        return theano.ifelse.ifelse(do_chart_update(x),
                (t,xp,chart),
                (t,T.stack((new_x,new_p)),new_chart)
            )

    M.Hamiltonian_dynamics = lambda q,p: integrate(ode_Hamiltonian,chart_update_Hamiltonian,T.stack((q[0],p)), q[1])
    M.Hamiltonian_dynamicsf = M.coords_function(M.Hamiltonian_dynamics,p)

    ## Geodesic
    def Exp_Hamiltonian(q,p):
        curve = M.Hamiltonian_dynamics(q,p)
        q = curve[1][-1,0]
        chart = curve[2][-1]
        return(q,chart)
    M.Exp_Hamiltonian = Exp_Hamiltonian
    M.Exp_Hamiltonianf = M.coords_function(M.Exp_Hamiltonian,p)
    def Exp_Hamiltoniant(q,p):
        curve = M.Hamiltonian_dynamics(q,p)
        q = curve[1][:,0]
        chart = curve[2]
        return(q,chart)
    M.Exp_Hamiltoniant = Exp_Hamiltoniant
    M.Exp_Hamiltoniantf = M.coords_function(M.Exp_Hamiltoniant,p)

