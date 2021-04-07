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

from src.manifolds.manifold import *

from src.plotting import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class Ellipsoid(EmbeddedManifold):
    """ 2d Ellipsoid """

    def _chart(self):
        """ return default coordinate chart """
        if self.chart_center == 'x':
            return T.eye(3)[:,0]
        elif self.chart_center == 'y':
            return T.eye(3)[:,1]
        elif self.chart_center == 'z':
            return T.eye(3)[:,2]
        else:
            assert(False)

    def _centered_chart(self,x):
        """ return centered coordinate chart """
        return x

    def get_B(self,v):
        """ R^3 basis with first basis vector v """
        b1 = v
        k = T.argmin(T.abs_(v))
        ek = T.eye(3)[:,k]
        b2 = ek-v[k]*v
        b3 = cross(b1,b2)
        return T.stack((b1,b2,b3),axis=1)

    # Logarithm with standard Riemannian metric on S^2
    def StdLog(self, x,y): 
        proj = lambda x,y: T.dot(x,y)*x
        Fx = self.F(x)
        v = y-proj(Fx,y)
        theta = T.arccos(T.dot(Fx,y))
        normv = T.nlinalg.norm(v,2)
        w = theano.ifelse.ifelse(T.ge(normv,1e-5),
                theta/normv*v,
                T.zeros_like(v)
            )
        return T.dot(self.invJF((Fx,x[1])),w)

    def __init__(self,params=np.array([1.,1.,1.]),chart_center='z',use_spherical_coords=False):
        self.params = theano.shared(np.array(params)) # ellipsoid parameters (e.g. [1.,1.,1.] for sphere)
        self.use_spherical_coords = use_spherical_coords
        self.chart_center = chart_center

        if not use_spherical_coords:
            F = lambda x: T.dot(self.get_B(x[1]),params*T.stack([-(-1+x[0][0]**2+x[0][1]**2),2*x[0][0],2*x[0][1]])/(1+x[0][0]**2+x[0][1]**2))
            def invF(x):
                Rinvx = T.slinalg.Solve()(self.get_B(x[1]),x[0])
                return T.stack([Rinvx[1]/(1+Rinvx[0]),Rinvx[2]/(1+Rinvx[0])])
            self.do_chart_update = lambda x: T.le(T.sum(T.square(x[0])),.05) # look for a new chart if false
        # spherical coordinates, no charts
        x = self.sym_coords() # Point on M in coordinates
        self.F_spherical = lambda phitheta: params*T.stack([T.sin(phitheta[1]-np.pi/2)*T.cos(phitheta[0]),T.sin(phitheta[1]-np.pi/2)*T.sin(phitheta[0]),T.cos(phitheta[1]-np.pi/2)])
        self.F_sphericalf = theano.function([x], self.F_spherical(x))
        self.JF_spherical = lambda x: T.jacobian(self.F_spherical(x),x)
        self.JF_sphericalf = theano.function([x], self.JF_spherical(x))
        self.F_spherical_inv = lambda x: T.stack([T.arctan2(x[1],x[0]),T.arccos(x[2])])
        self.F_spherical_invf = theano.function([x], self.F_spherical_inv(x))
        self.g_spherical = lambda x: T.dot(self.JF_spherical(x).T,self.JF_spherical(x))
        self.mu_Q_spherical = lambda x: 1./T.nlinalg.Det()(self.g_spherical(x))
        self.mu_Q_sphericalf = theano.function([x],self.mu_Q_spherical(x))

        ## optionally use spherical coordinates in chart computations
        #if use_spherical_coords:
        #    F = lambda x: T.dot(x[1],self.F_spherical(x[0]))

        EmbeddedManifold.__init__(self,F,2,3,invF=invF)
        self.chart = self._chart
        self.centered_chart = self._centered_chart
        self.chartf = theano.function([],self.chart())
        self.centered_chartf = self.function(self.centered_chart)

        ## hardcoded Jacobian for speed (removing one layer of differentiation)
        #if not use_spherical_coords:
        #    self.JF = lambda x: T.stack(params)[:,np.newaxis]*T.stack([-2*x[0]**2+2*x[1]**2+2,-4*x[0]*x[1],-4*x[0]*x[1],2*x[0]**2-2*x[1]**2+2,-4*x[0],-4*x[1]]).reshape((3,2))/(1+x[0]**2+x[1]**2)**2
        #else:
        #    self.JF = lambda phitheta: T.stack(params)[:,np.newaxis]*T.stack([-T.sin(phitheta[0])*T.sin(phitheta[1]),T.cos(phitheta[0])*T.cos(phitheta[1]),T.cos(phitheta[0])*T.sin(phitheta[1]),T.sin(phitheta[0])*T.cos(phitheta[1]),0,-T.sin(phitheta[1])]).reshape((3,2))

        #self.JF_theanof = self.JFf
        #self.JFf = self.function(self.JF)

        # metric matrix
        x = self.sym_coords()
        self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))

        # action of matrix group on elements
        x = self.sym_element()
        g = T.matrix() # group matrix
        gs = T.tensor3() # sequence of matrices
        self.act = lambda g,x: T.tensordot(g,x,(1,0))
        self.acts = lambda g,x: T.tensordot(g,x,(2,0))
        self.actf = theano.function([g,x], self.act(g,x))
        self.actsf = theano.function([gs,x], self.acts(gs,x))

        # Logarithm with standard Riemannian metric on S^2        
        self.StdLogf = self.coords_function(self.StdLog,self.sym_element())

    def __str__(self):
        return "%dd ellipsoid, parameters %s, spherical coords %s" % (self.dim.eval(),self.params.eval(),self.use_spherical_coords)

    def newfig(self):
        newfig3d()

    def plot(self,rotate=None,alpha=None,lw=0.3):
        ax = plt.gca(projection='3d')
        x = np.arange(-10,10,1)
        ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.xaxis._axinfo["grid"]['linewidth'] = lw
        ax.yaxis._axinfo["grid"]['linewidth'] = lw
        ax.zaxis._axinfo["grid"]['linewidth'] = lw
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_zlim(-1.,1.)
        #ax.set_aspect("equal")
        if rotate is not None:
            ax.view_init(rotate[0],rotate[1])
    #     else:
    #         ax.view_init(35,225)
        plt.xlabel('x')
        plt.ylabel('y')

    #    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #draw ellipsoid
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=self.params.eval()[0]*np.cos(u)*np.sin(v)
        y=self.params.eval()[1]*np.sin(u)*np.sin(v)
        z=self.params.eval()[2]*np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)

        if alpha is not None:
            ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)


    def plot_field(self, field,lw=.3):
        ax = plt.gca(projection='3d')
        x = np.arange(-10,10,1)
        ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.xaxis._axinfo["grid"]['linewidth'] = lw
        ax.yaxis._axinfo["grid"]['linewidth'] = lw
        ax.zaxis._axinfo["grid"]['linewidth'] = lw
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_zlim(-1.,1.)
        #ax.set_aspect("equal")

        plt.xlabel('x')
        plt.ylabel('y')

#        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #draw ellipsoid
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x=self.params.eval()[0]*np.cos(u)*np.sin(v)
        y=self.params.eval()[1]*np.sin(u)*np.sin(v)
        z=self.params.eval()[2]*np.cos(v)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Fx = np.array([x[i,j],y[i,j],z[i,j]])
                chart = self.centered_chartf(Fx)
                xcoord = self.invFf((Fx,chart))
                v = field((xcoord,chart))
                self.plotx((xcoord,chart),v=v)
