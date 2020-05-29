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

class Cylinder(EmbeddedManifold):
    """ 2d Cylinder """

    def _chart(self):
        """ return default coordinate chart """
        return T.zeros(self.dim)

    def _centered_chart(self,coords=None):
        """ return centered coordinate chart """
        return self.invF((coords,self.chart()))  # chart centered at coords

    def get_B(self,v):
        """ R^3 basis with first basis vector v """
        b1 = v
        k = T.argmin(T.abs_(v))
        ek = T.eye(3)[:,k]
        b2 = ek-v[k]*v
        b3 = cross(b1,b2)
        return T.stack((b1,b2,b3),axis=1)

    # Logarithm with standard Riemannian metric
    def StdLog(_x,y): 
        self = M
        x = self.invF((self.F(_x),_x[1]))
        y = self.invF((y,_x[1]))
        return self.radius*(y-x)
    
    def __init__(self,params=(1.,np.array([0.,1.,0.]),0.)):
        self.radius = theano.shared(tensor(params[0])) # axis of cylinder
        self.orientation = theano.shared(tensor(params[1])) # axis of cylinder
        self.theta = theano.shared(tensor(params[2])) # angle around rotation axis

        F = lambda x: T.dot(self.get_B(self.orientation),
                T.stack([x[0][1]+x[1][1],self.radius*T.cos(self.theta+x[1][0]+x[0][0]),self.radius*T.sin(self.theta+x[1][0]+x[0][0])]))
        def invF(x):
            Rinvx = T.slinalg.Solve()(self.get_B(self.orientation),x[0])
            rotangle = -(self.theta+x[1][0])
            rot = T.dot(T.stack(
                (T.stack((T.cos(rotangle),-T.sin(rotangle))),
                 T.stack((T.sin(rotangle),T.cos(rotangle))))),
                Rinvx[1:])
            return T.stack([T.arctan2(rot[1],rot[0]),Rinvx[0]-x[1][1]])
        self.do_chart_update = lambda x: T.le(T.abs_(x[0][0]),np.pi/2) # look for a new chart if false

        EmbeddedManifold.__init__(self,F,2,3,invF=invF)
        self.chart = self._chart
        self.centered_chart = self._centered_chart
        self.chartf = theano.function([],self.chart())
        self.centered_chartf = self.function(self.centered_chart)

        # metric matrix
        x = self.sym_coords()
        self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))

    def __str__(self):
        return "cylinder in R^3, radius %s, axis %s, rotation around axis %s" % (self.radius.eval(),self.orientation.eval(),self.theta.eval())

    def newfig(self):
        newfig3d()

    def plot(self, rotate=None,alpha=None,lw=0.3):
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
        u, v = np.mgrid[-np.pi:np.pi:20j, -np.pi:np.pi:10j]
        x = np.zeros(u.shape)
        y = np.zeros(u.shape)
        z = np.zeros(u.shape)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                w = self.Ff(self.coordsf(tensor([u[i,j],v[i,j]])))
                x[i,j] = w[0]; y[i,j] = w[1]; z[i,j] = w[2]
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)

        if alpha is not None:
            ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)

    # plot path
    def plot_path(self, xs, u=None, v=None, N_vec=np.arange(0,n_steps.eval()), i0=0, color='b', color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None, v=v[i] if v is not None else None,
                       N_vec=N_vec,i0=i,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==(N-1))
            prevx = x
        return
    
    
    # plot x. x can be either in coordinates or in R^3
    def plotx(self, x, u=None, v=None, N_vec=np.arange(0,n_steps.eval()), i0=0, color='b', color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
        
        assert(type(x) == type(()) or x.shape[0] == self.emb_dim.eval())

        if type(x) == type(()): # map to S2
            Fx = self.Ff(x)
            chart = x[1]
        else: # get coordinates
            Fx = x
            chart = self.centered_chartf(Fx)
            x = (self.invFf((Fx,chart)),chart)
          
        if prevx is not None:
            if type(prevx) == type(()): # map to S2
                Fprevx = self.Ff(prevx)
            else:
                Fprevx = prevx
                prevx = (self.invFf((Fprevx,chart)),chart)
    
        ax = plt.gca(projection='3d')
        if prevx is None or last:
            ax.scatter(Fx[0],Fx[1],Fx[2],color=color,s=s)
        if prevx is not None:
            xx = np.stack((Fprevx,Fx))
            ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
    
        if u is not None:
            Fu = np.dot(self.JFf(x), u)
            ax.quiver(Fx[0], Fx[1], Fx[2], Fu[0], Fu[1], Fu[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                      color='black')
    
        if v is not None:
            if i0 in N_vec:
                v = np.dot(self.JFf(x), v)
                ax.quiver(Fx[0], Fx[1], Fx[2], v[0], v[1], v[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                        color='black')

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

        u, v = np.mgrid[-np.pi:np.pi:40j, -np.pi:np.pi:20j]
        x = np.zeros(u.shape)
        y = np.zeros(u.shape)
        z = np.zeros(u.shape)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                w = self.Ff(self.coordsf(tensor([u[i,j],v[i,j]])))
                x[i,j] = w[0]; y[i,j] = w[1]; z[i,j] = w[2]
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Fx = np.array([x[i,j],y[i,j],z[i,j]])
                chart = self.centered_chartf(Fx)
                xcoord = self.invFf((Fx,chart))
                v = field((xcoord,chart))
                self.plotx((xcoord,chart),v=v)



