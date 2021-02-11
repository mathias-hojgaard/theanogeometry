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

import matplotlib.pyplot as plt

class Manifold(object):
    """ Base manifold class """

    def __init__(self):
        self.dim = None

    def __init__(self):
        self.rank = None

    def sym_element(self):
        """ return symbolic element in manifold """
        return T.vector()

    def sym_elements(self):
        """ return symbolic sequence of elements in manifold """
        return T.matrix()

    def sym_coords(self):
        """ return symbolic coordinate representation of point in manifold """
        return T.vector()

    def sym_chart(self):
        """ return symbolic specified coordinate chart """
        return T.vector()

    def sym_charts(self):
        """ return symbolic specified coordinate charts """
        return T.matrix()

    def chart(self):
        """ return default or specified coordinate chart. This method will generally be overriding by inheriting classes """
        # default value (must be a vector in accordance with type of sym_chart())
        return T.zeros(1) 

    def centered_chart(self,coords):
        """ return centered coordinate chart. Must be implemented by inheriting classes """
        assert(False)

    def coordsf(self,coords=None,chart=None):
        """ return coordinate representation of point in manifold """
        if coords is None:
            coords = tensor(np.zeros(self.dim.eval()))
        if chart is None:
            chart = self.chartf()

        return (tensor(coords),chart)

    def update_coords(self,coords,new_chart):
        """ change between charts """
        assert(False) # not implemented here

    def update_vector(self,coords,new_coords,v):
        """ change tangent vector between charts """
        assert(False) # not implemented here

    def update_covector(self,coords,new_coords,p):
        """ change cotangent vector between charts """
        assert(False) # not implemented here

    def sym_vector(self):
        """ return symbolic tangent vector """
        return T.vector()

    def sym_vectors(self):
        """ return symbolic sequence of tangent vector """
        return T.matrix()

    def sym_covector(self):
        """ return symbolic cotangent vector """
        return T.vector()

    def sym_coordsvector(self):
        """ return symbolic tangent vector in coordinate representation """
        return T.vector()

    def sym_coordscovector(self):
        """ return symbolic cotangent vector in coordinate representation """
        return T.vector()

    def sym_frame(self):
        """ return symbolic frame for tangent space """
        return T.matrix()

    def sym_process(self):
        """ return symbolic steps of process """
        return T.matrix()

    def coords_function(self,f,*args):
        """ compile function on manifold. Result function takes coordinates and chart + optinal parameters """

        coords = self.sym_coords()
        chart = self.sym_chart()
        _f = theano.function([coords,chart]+list(args),f((coords,chart),*args),on_unused_input='ignore')

        def ff(x,*args):
            if type(x) == type(()):
                xargs = x+args
            elif type(x) == type([]):
                xargs = tuple(x)+args
            else:
                xargs = (x,)+args

            return _f(*xargs)

        return ff

    def function(self,f,*args):
        """ compile function on manifold. Result function takes element on manifold + optinal parameters """

        x = self.sym_element()
        ff = theano.function([x,]+list(args),f(x,*args),on_unused_input='ignore')
             
        return ff

    def newfig(self):
        """ open new plot for manifold """

    def __str__(self):
        return "abstract manifold"

class EmbeddedManifold(Manifold):
    """ Embedded manifold base class """

    def _update_coords(self,coords,new_chart):
        """ change between charts """
        return (self.invF((self.F(coords),new_chart)),new_chart)

    def _update_vector(self,coords,new_coords,new_chart,v):
        """ change tangent vector between charts """
        return T.dot(self.invJF((self.F((new_coords,new_chart)),new_chart)),T.dot(self.JF(coords),v))

    def _update_covector(self,coords,new_coords,new_chart,p):
        """ change cotangent vector between charts """
        return T.dot(self.JF((new_coords,new_chart)).T,T.dot(self.invJF((self.F(coords),coords[1])).T,p))

    def __init__(self,F=None,dim=None,emb_dim=None,invF=None):
        Manifold.__init__(self)
        self.dim = None if dim is None else constant(dim)
        self.emb_dim = None if dim is None else constant(emb_dim)

        # embedding map and its inverse
        if F is not None:
            self.F = F
            self.invF = invF
            self.Ff = self.coords_function(self.F)
            self.JF = lambda x: T.jacobian(self.F(x),x[0])
            self.JFf = self.coords_function(self.JF)
            if invF != None:
                self.invFf = self.coords_function(self.invF)
                self.invJF = lambda x: T.jacobian(self.invF(x),x[0])
                self.invJFf = self.coords_function(self.invJF)

            self.update_coords = self._update_coords
            self.update_vector = self._update_vector
            self.update_covector = self._update_covector
            new_chart = self.sym_chart()
            self._update_coordsf = self.coords_function(self.update_coords,new_chart)
            self.update_coordsf = lambda coords,new_chart: tuple(self._update_coordsf(coords,new_chart))
            new_coords = self.sym_coords()
            v = self.sym_vector()
            self._update_vectorf = self.coords_function(self.update_vector,new_coords,new_chart,v)
            self.update_vectorf = lambda x,new_coords,v: self._update_vectorf(x,new_coords[0],new_coords[1],v)
            p = self.sym_covector()
            self._update_covectorf = self.coords_function(self.update_covector,new_coords,new_chart,p)
            self.update_covectorf = lambda x,new_coords,p: self._update_covectorf(x,new_coords[0],new_coords[1],p)

            # metric matrix
            self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))

            ## get coordinate representation from embedding space
            #from scipy.optimize import minimize
            #def get_get_coords():
            #    x = self.sym_coords()
            #    y = self.sym_coords()
            #    chart = self.sym_coords()
            #
            #    loss = lambda x,y,chart: 1./self.emb_dim.eval()*T.sum(T.sqr(self.F((x,chart))-y))
            #    dloss = lambda x,y,chart: T.grad(loss(x,y,chart),x)
            #    dlossf = theano.function([x,y,chart], (loss(x,y,chart),dloss(x,y,chart)))
            #
            #    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
            #    def get_coords(y,x0=None,chart=None):        
            #        if chart is None:
            #            chart = self.chart()
            #        def f(x):
            #            (z,dz) = dlossf(x,tensor(y),chart)
            #            return (z.astype(np.double),dz.astype(np.double))
            #        if x0 is None:
            #            x0 = np.zeros(self.dim.eval()).astype(np.double)
            #        res = minimize(f, x0, method='CG', jac=True, options={'disp': False, 'maxiter': 100})
            #        return res.x
            #    
            #    return get_coords
            #self.get_coordsf = get_get_coords()
            
    # plot path
    def plot_path(self, xs, u=None, vs=None, v_steps=None, i0=0, color='b', 
                  color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
        
        if vs is not None and v_steps is not None:
            v_steps = np.arange(0,n_steps.eval())
        
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            xx = x[0] if type(x) is tuple else x
            if xx.shape > self.dim.eval() and (self.emb_dim == None or xx.shape != self.emb_dim.eval()): # attached vectors to display
                v = xx[self.dim.eval():].reshape((self.dim.eval(),-1))
                x = (xx[0:self.dim.eval()],x[1]) if type(x) is tuple else xx[0:self.dim.eval()]
            elif vs is not None:
                v = vs[i]
            else:
                v = None
            self.plotx(x, u=u if i == 0 else None, v=v,
                       v_steps=v_steps,i=i,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==(N-1))
            prevx = x 
        return

    # plot x. x can be either in coordinates or in R^3
    def plotx(self, x, u=None, v=None, v_steps=None, i=0, color='b',               
              color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):

        assert(type(x) == type(()) or x.shape[0] == self.emb_dim.eval())
        
        if v is not None and v_steps is None:
            v_steps = np.arange(0,n_steps.eval())        

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
            if i in v_steps:
                v = np.dot(self.JFf(x), v)
                ax.quiver(Fx[0], Fx[1], Fx[2], v[0], v[1], v[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                        color='black')

    def __str__(self):
        return "dim %d manifold embedded in R^%d" % (self.dim.eval(),self.emb_dim.eval())
