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
    """ Brownian motion from stochastic development """

    x = M.sym_element()
    u = M.sym_FM_element()
    
    def Brownian_development(x,dWt):
        # amend x with orthogonal basis to get initial frame bundle element
        gsharpx = M.gsharp(x)
        nu = theano.tensor.slinalg.Cholesky()(gsharpx)
        u = (T.concatenate((x[0],nu.flatten())),x[1])
        
        (ts,us,charts) = M.stochastic_development(u,dWt)
        
        return (ts,us[:,0:M.dim],charts)
    
    M.Brownian_development = Brownian_development
    M.Brownian_developmentf = M.coords_function(M.Brownian_development,dWt) 
