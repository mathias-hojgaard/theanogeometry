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
from src.params import *

from src.groups.group import *

from src.linalg import *
import theano.tensor.slinalg

import matplotlib.pyplot as plt

class GLN(LieGroup):
    """ General linear group GL(N) """

    def __init__(self,N=3):
        LieGroup.__init__(self,N=N,invariance='left')

        self.dim = constant(N*N) # group dimension

        # project to group, here with minimum eigenvalue 1e-3
        def to_group(g):
            _min_eig = 1e-3
            w, V = T.nlinalg.eig(g.astype('complex128'))
            w_prime = T.where(abs(w) < _min_eig, _min_eig, w)
            return T.dot(V,T.dot(T.diag(w_prime),V.T)).real
        g = self.sym_element()
        self.to_groupf = self.function(to_group)

        ## coordinate chart on the linking Lie algebra, trival in this case
        def VtoLA(hatxi): # from \RR^G.dim to LA
            if hatxi.type == T.vector().type:
                return hatxi.reshape((N,N))
            else: # matrix
                return hatxi.reshape((N,N,-1))
        self.VtoLA = VtoLA
        def LAtoV(m): # from LA to \RR^G.dim
            if m.type == T.matrix().type:
                return m.reshape((self.dim,))
            elif m.type == T.tensor3().type:
                return m.reshape((self.dim,-1))
            else:
                assert(False)
        self.LAtoV = LAtoV

        self.Expm = T.slinalg.Expm()
        #Expm = linalg.Expm()
        self.Logm = lambda g : Logm()(g)

        super(GLN,self).initialize()
        
    def __str__(self):
        return "GL(%d) (dimension %d)" % (self.N.eval(),self.dim.eval())        

    def plot_path(self, g,color_intensity=1.,color=None,linewidth=3., alpha = 0.1,prevg=None):
        assert(len(g.shape)>2)
        for i in range(g.shape[0]):
            self.plotg(g[i],
                  linewidth=linewidth if i==0 or i==g.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==g.shape[0]-1 else .7,
                  alpha = alpha,
                  prevg=g[i-1] if i>0 else None)
        return

    def plotg(self, g,color_intensity=1.,color=None,linewidth=3., alpha = 0.1,prevg=None):
        s0 = np.eye(self.N.eval()) # shape
        s = np.dot(g,s0) # rotated shape
        if prevg is not None:
            prevs = np.dot(prevg,s0)

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        for i in range(s.shape[1]):
            plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1, alpha=alpha)
            if prevg is not None:
                ss = np.stack((prevs,s))
                plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
