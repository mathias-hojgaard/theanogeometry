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
    """ development and stochastic development from R^d to M """

    t = T.scalar()
    dgammas = T.matrix() # velocity of Euclidean curve
    dsm = T.matrix() # derivative of Euclidean semimartingale
    u = M.sym_FM_element()
    d = M.dim

    # Deterministic development
    def ode_development(dgamma,t,u,chart):
        nu = u[d:].reshape((d,-1))
        m = nu.shape[1]

        det = T.tensordot(M.Horizontal((u,chart))[:,0:m], dgamma, axes = [1,0])
    
        return det

    M.development = lambda u,dgammas: integrate(ode_development,M.chart_update_FM,u[0],u[1],dgammas)
    M.developmentf = M.coords_function(M.development,dgammas)

    # Stochastic development
    def sde_development(dsm,t,u,chart):
        nu = u[d:].reshape((d,-1))
        m = nu.shape[1]

        Hu = M.Horizontal((u,chart))
        
        sto = T.tensordot(Hu[:,0:m], dsm, axes = [1,0])
    
        return (T.zeros_like(sto), sto, Hu[:,0:m])

    M.sde_development = sde_development
    M.stochastic_development = lambda u,dsm: integrate_sde(sde_development,integrator_stratonovich,M.chart_update_FM,u[0],u[1],dsm)
    M.stochastic_developmentf = M.coords_function(M.stochastic_development,dsm)