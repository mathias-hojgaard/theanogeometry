from src.setup import *
from src.params import *
from src.group import *
from src.plotting import *
import src.linalg as linalg

##########################################################################
# this file contains definitions for SPD(N)                              #
##########################################################################

manifold = "SPD(N)"

d = N*N
M_emb_dim = d

# action of matrix group on elements
g = T.matrix() # \RR^{NxN} matrix
gs = T.tensor3() # sequence of \RR^{NxN} matrices
def act(g,q):
    if g.type == T.matrix().type:
        return T.tensordot(g,T.tensordot(q.reshape((N,N)),g,(1,1)),(1,0))
    elif g.type == T.tensor3().type: # list of matrices
        (cout, updates) = theano.scan(fn=lambda g,x: T.tensordot(g,T.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)),
        outputs_info=[T.eye(N)],
        sequences=[g.dimshuffle((2,0,1))])
        
        return cout.dimshuffle((1,2,0))
    else:
        assert(False)
actf = theano.function([g,q], act(g,q))
actsf = theano.function([gs,q], act(gs,q))

### Plotting:
#########################

import matplotlib.pyplot as plt

def plotM(rotate=None, alpha = None):
    ax = plt.gca(projection='3d')
    ax.set_aspect("equal")
    if rotate != None:
        ax.view_init(rotate[0],rotate[1])
#     else:
#         ax.view_init(35,225)
    plt.xlabel('x')
    plt.ylabel('y')

def plotx(x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
    if len(x.shape)>2:
        for i in range(x.shape[0]):
            plotx(x[i],
                  linewidth=linewidth if i==0 or i==x.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                  prevx=x[i-1] if i>0 else None,ellipsoid=ellipsoid,i=i,maxi=x.shape[0])
        return
    (w,V) = np.linalg.eigh(x)
    s = np.sqrt(w[np.newaxis,:])*V # scaled eigenvectors
    if prevx is not None:
        (prevw,prevV) = np.linalg.eigh(prevx)
        prevs = np.sqrt(prevw[np.newaxis,:])*prevV # scaled eigenvectors
        ss = np.stack((prevs,s))

    colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
    if ellipsoid is None:
        for i in range(s.shape[1]):
            plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
            if prevx is not None:
                plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
    else:
        try:
            if i % int(ellipsoid['step']) != 0 and i != maxi-1:
                return
        except:
            pass
        try:
            if ellipsoid['subplot']:
                (fig,ax) = newfig(1,maxi//int(ellipsoid['step'])+1,i//int(ellipsoid['step'])+1,new_figure=i==0)
        except:
            ax = plot.gca(projection='3d')
        #draw ellipsoid, from https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
        U, ss, rotation = np.linalg.svd(x)
        radii = np.sqrt(ss)
        u = np.linspace(0., 2.*np.pi, 20)
        v = np.linspace(0., np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for l in range(x.shape[0]):
            for k in range(x.shape[1]):
                [x[l,k],y[l,k],z[l,k]] = np.dot([x[l,k],y[l,k],z[l,k]], rotation)
        ax.plot_surface(x, y, z, facecolors=cm.winter(y/np.amax(y)), linewidth=0, alpha=ellipsoid['alpha'])
        for i in range(s.shape[1]):
            plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)        
        plt.axis('off')
