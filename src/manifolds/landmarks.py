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

import matplotlib.pyplot as plt

class landmarks(Manifold):
    """ LDDMM landmark manifold """

    def __init__(self,N=1,m=2,k_alpha=1.,k_sigma=np.diag((.5,.5)),kernel='Gaussian'):
        Manifold.__init__(self)

        self.N = theano.shared(N) # number of landmarks
        self.m = T.constant(m) # landmark space dimension (usually 2 or 3
        self.dim = self.m*self.N
        self.rank = theano.shared(self.dim.eval())

        self.chartf = theano.function([],self.chart())
        self.update_coords = lambda coords,_: coords
        new_chart = self.sym_chart()
        self._update_coordsf = self.coords_function(self.update_coords,new_chart)
        self.update_coordsf = lambda coords,new_chart: tuple(self._update_coordsf(coords,new_chart))

        self.k_alpha = theano.shared(scalar(k_alpha))
        self.k_sigma = theano.shared(tensor(k_sigma)) # standard deviation of the kernel
        self.inv_k_sigma = theano.tensor.nlinalg.MatrixInverse()(self.k_sigma)
        self.k_Sigma = T.tensordot(self.k_sigma,self.k_sigma,(1,1))
        self.kernel = kernel

        ##### Kernel on M:
        if self.kernel == 'Gaussian':
            k = lambda x: self.k_alpha*T.exp(-.5*T.sqr(T.tensordot(x,self.inv_k_sigma,(0 if x.type == T.vector().type else 2,1))).sum(0 if x.type == T.vector().type else 2))
        elif self.kernel == 'K1':
            def k(x):
                r = T.sqrt((1e-7+T.sqr(T.tensordot(x,self.inv_k_sigma,(0 if x.type == T.vector().type else 2,1))).sum(0 if x.type == T.vector().type else 2)))
                return self.k_alpha*2*(1+r)*T.exp(-r)
        elif self.kernel == 'K2':
            def k(x):
                r = T.sqrt((1e-7+T.sqr(T.tensordot(x,self.inv_k_sigma,(0 if x.type == T.vector().type else 2,1))).sum(0 if x.type == T.vector().type else 2)))
                return self.k_alpha*4*(3+3*r+r**2)*T.exp(-r)
        elif self.kernel == 'K3':
            def k(x):
                r = T.sqrt((1e-7+T.sqr(T.tensordot(x,self.inv_k_sigma,(0 if x.type == T.vector().type else 2,1))).sum(0 if x.type == T.vector().type else 2)))
                return self.k_alpha*8*(15+15*r+6*r**2+r**3)*T.exp(-r)
        elif self.kernel == 'K4':
            def k(x):
                r = T.sqrt((1e-7+T.sqr(T.tensordot(x,self.inv_k_sigma,(0 if x.type == T.vector().type else 2,1))).sum(0 if x.type == T.vector().type else 2)))
                return self.k_alpha*16*(105+105*r+45*r**2+10*r**3+r**4)*T.exp(-r)
        else:
            raise Exception('unknown kernel specified')
        self.k = k
        dk = lambda x: T.grad(k(x),x)
        self.dk = dk
        d2k = lambda x: theano.gradient.hessian(k(x),x)
        self.d2k = d2k

        # in coordinates
        q1 = self.sym_element()
        q2 = self.sym_element()
        self.k_q = lambda q1,q2: self.k(q1.reshape((-1,m)).dimshuffle(0,'x',1)-q2.reshape((-1,m)).dimshuffle('x',0,1))
        self.k_qf = theano.function([q1,q2],self.k_q(q1,q2))
        self.K = lambda q1,q2: (self.k_q(q1,q2)[:,:,np.newaxis,np.newaxis]*T.eye(self.m)[np.newaxis,np.newaxis,:,:]).dimshuffle((0,2,1,3)).reshape((-1,self.dim))
        self.Kf = theano.function([q1,q2],self.K(q1,q2))

        ##### Metric:
        def gsharp(q):
            return self.K(q[0],q[0])
        self.gsharp = gsharp


        #### landmark specific setup (see Micheli, Michor, Mumford 2013)
        self.dK = lambda q1,q2: T.jacobian(self.K(q1,theano.gradient.disconnected_grad(q2)).flatten(),q1).reshape((self.N,self.m,self.N,self.m,self.N,self.m))
        self.d2K = lambda q1,q2: T.jacobian(self.DK(q1,theano.gradient.disconnected_grad(q2)).flatten(),q1).reshape((self.N,self.m,self.N,self.m,self.N,self.m,self.N,self.m))
        #self.P = lambda q1,q2,alpha,beta: self.dK(q1,q2)

    def __str__(self):
        return "%d landmarks in R^%d (dim %d). kernel %s, k_alpha=%d, k_sigma=%s" % (self.N.eval(),self.m.eval(),self.dim.eval(),self.kernel,self.k_alpha.eval(),self.k_sigma.eval())

    def plot(self):
        plt.axis('equal')

    def plot_path(self, xs, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, curve=False, markersize=None, arrowcolor='k'):
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       prevx=prevx,
                       last=i==N-1,
                       curve=curve)
            prevx = x
        return

    def plotx(self, x, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, curve=False, markersize=None, arrowcolor='k'):
        assert(type(x) == type(()) or x.shape[0] == self.dim.eval())
        if type(x) == type(()):
            x = x[0]
        if type(prevx) == type(()):
            prevx = prevx[0]

        x = x.reshape((-1,self.m.eval()))
        NN = x.shape[0]

        for j in range(NN):
            if last:
                plt.scatter(x[j,0],x[j,1],color=color,s=markersize)
            else:
                try:
                    prevx = prevx.reshape((NN,self.m.eval()))
                    xx = np.stack((prevx[j,:],x[j,:]))
                    plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
                except:
                    plt.scatter(x[j,0],x[j,1],color=color,s=markersize)

            try:
                u = u.reshape((NN, self.m.eval()))
                plt.quiver(x[j,0], x[j,1], u[j, 0], u[j, 1], pivot='tail', linewidth=linewidth, scale=5, color=arrowcolor)
            except:
                pass
        if curve and (last or prevx is None):
            plt.plot(np.hstack((x[:,0],x[0,0])),np.hstack((x[:,1],x[0,1])),'o-',color=color)

#    # plot point in frame bundle FM
#    def plotFMx(self,u,N_vec=0,i0=0,color=np.array(['g','b']),s=10,color_intensity=1.,linewidth=3.,prevx=None,last=True):
#        if len(u.shape)>1:
#            for i in range(u.shape[0]):
#                self.plotFMx(u[i],
#                        N_vec=N_vec,i0=i,               
#                        color=color, s=s,        
#                        linewidth=linewidth if i==0 or i==u.shape[0]-1 else .8,
#                        color_intensity=color_intensity if i==0 or i==u.shape[0]-1 else .7,
#                        prevx=u[i-1] if i>0 else None,
#                        last=i==(u.shape[0]-1))                  
#            return
#    
#        x = u[0:self.dim.eval()].reshape((self.N.eval(),self.m.eval()))
#        nu = u[self.dim.eval():].reshape((self.N.eval(),self.m.eval(),-1))
#     
#        for j in range(self.N.eval()):
#            if prevx == None or last:
#                plt.scatter(x[j,0],x[j,1],color=color[0])
#            if prevx != None:
#                prevxx = prevx[0:self.dim.eval()].reshape((self.N.eval(),2))
#                xx = np.stack((prevxx[j,:],x[j,:]))
#                plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color[1])
#    
#            if N_vec != None:                                              
#               Seq = lambda m, n: [t*n//m + n//(2*m) for t in range(m)]
#               Seqv = np.hstack((0,Seq(N_vec,n_steps.eval())))
#               if i0 == 0 or i0 in Seqv:
#                   for k in range(nu.shape[2]):
#                       plt.quiver(x[j,0],x[j,1],nu[j,0,k],nu[j,1,k],pivot='tail',linewidth=linewidth,scale=s)


    # grid plotting functions
    import itertools

    """
    Example usage:
    (grid,Nx,Ny)=getGrid(-1,1,-1,1,xpts=50,ypts=50)
    plotGrid(grid,Nx,Ny)
    """

    def d2zip(self,grid):
        return np.dstack(grid).reshape([-1,2])

    def d2unzip(self,points,Nx,Ny):
        return np.array([points[:,0].reshape(Nx,Ny),points[:,1].reshape(Nx,Ny)])

    def getGrid(self,xmin,xmax,ymin,ymax,xres=None,yres=None,xpts=None,ypts=None):
        """
        Make regular grid
        Grid spacing is determined either by (x|y)res or (x|y)pts
        """

        if xres:
            xd = xres
        elif xpts:
            xd = np.complex(0,xpts)
        else:
            assert(False)
        if yres:
            yd = yres
        elif ypts:
            yd = np.complex(0,ypts)
        else:
            assert(False)

        grid = np.mgrid[xmin:xmax:xd,ymin:ymax:yd]
        Nx = grid.shape[1]
        Ny = grid.shape[2]

        return (self.d2zip(grid),Nx,Ny)


    def plotGrid(self,grid,Nx,Ny,coloring=True):
        """
        Plot grid
        """

        xmin = grid[:,0].min(); xmax = grid[:,0].max()
        ymin = grid[:,1].min(); ymax = grid[:,1].max()
        border = .5*(0.2*(xmax-xmin)+0.2*(ymax-ymin))

        grid = self.d2unzip(grid,Nx,Ny)

        color = 0.75
        colorgrid = np.full([Nx,Ny],color)
        cm = plt.cm.get_cmap('gray')
        if coloring:
            cm = plt.cm.get_cmap('coolwarm')
            hx = (xmax-xmin) / (Nx-1)
            hy = (ymax-ymin) / (Ny-1)
            for i,j in itertools.product(range(Nx),range(Ny)):
                p = grid[:,i,j]
                xs = np.empty([0,2])
                ys = np.empty([0,2])
                if 0 < i:
                    xs = np.vstack((xs,grid[:,i,j]-grid[:,i-1,j],))
                if i < Nx-1:
                    xs = np.vstack((xs,grid[:,i+1,j]-grid[:,i,j],))
                if 0 < j:
                    ys = np.vstack((ys,grid[:,i,j]-grid[:,i,j-1],))
                if j < Ny-1:
                    ys = np.vstack((ys,grid[:,i,j+1]-grid[:,i,j],))

                Jx = np.mean(xs,0) / hx
                Jy = np.mean(ys,0) / hy
                J = np.vstack((Jx,Jy,)).T

                A = .5*(J+J.T)-np.eye(2)
                CSstrain = np.log(np.trace(A*A.T))
                logdetJac = np.log(sp.linalg.det(J))
                colorgrid[i,j] = logdetJac

            cmin = np.min(colorgrid)
            cmax = np.max(colorgrid)
            f = 2*np.max((np.abs(cmin),np.abs(cmax),.5))
            colorgrid = colorgrid / f + 0.5

            print("mean color: ", np.mean(colorgrid))

        # plot lines
        for i,j in itertools.product(range(Nx),range(Ny)):
            if i < Nx-1:
                plt.plot(grid[0,i:i+2,j],grid[1,i:i+2,j],color=cm(colorgrid[i,j]))
            if j < Ny-1:
                plt.plot(grid[0,i,j:j+2],grid[1,i,j:j+2],color=cm(colorgrid[i,j]))

        #for i in range(0,grid.shape[1]):
        #    plt.plot(grid[0,i,:],grid[1,i,:],color)
        ## plot x lines
        #for i in range(0,grid.shape[2]):
        #    plt.plot(grid[0,:,i],grid[1,:,i],color)


        plt.xlim(xmin-border,xmax+border)
        plt.ylim(ymin-border,ymax+border)


    ### Misc
    def ellipse(self, cent, Amp):
        return  np.vstack(( Amp[0]*np.cos(np.linspace(0,2*np.pi*(1-1./self.N.eval()),self.N.eval()))+cent[0], Amp[1]*np.sin(np.linspace(0,2*np.pi*(1-1./self.N.eval()),self.N.eval()))+cent[1] )).T


