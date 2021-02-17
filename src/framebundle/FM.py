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
#from src.manifold import *
#from src.metric import *
from src.utils import *


def initialize(M,do_chart_update=None):
    """ Frame Bundle geometry """
    
    d  = M.dim

    x = M.sym_element()
    x1 = M.sym_element()
    v = M.sym_vector()
    nu = M.sym_frame()

    def sym_FM_element():
        """ return element of FM as concatenation (x,nu) flattened """
        return T.vector()
    def sym_FM_vector():
        """ vector in TFM """
        return T.vector()
    def sym_FM_covector():
        """ covector in T^*FM """
        return T.vector()
    M.sym_FM_element = sym_FM_element
    M.sym_FM_vector = sym_FM_vector
    M.sym_FM_covector = sym_FM_covector  

    u = M.sym_FM_element()
    q = M.sym_FM_element()
    w = M.sym_FM_vector()    
    p = M.sym_FM_covector()

    def chart_update_FM(t,u,chart,*args):
        if do_chart_update is None:
            return (t,u,chart)
        
        x = (u[0:d],chart)
        nu = u[d:].reshape((d,-1))

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_nu = M.update_vector(x,new_x,new_chart,nu)
        
        return theano.ifelse.ifelse(do_chart_update(x),
                (t,u,chart),
                (t,T.concatenate((new_x,new_nu.flatten())),new_chart)
            )
    M.chart_update_FM = chart_update_FM        

    #### Bases shifts, see e.g. Sommer Entropy 2016 sec 2.3
    # D denotes frame adapted to the horizontal distribution
    def to_D(u,w):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        wx = w[0:d]
        wnu = w[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = T.tensordot(M.Gamma_g(x),nu,(2,0)).dimshuffle(0,2,1)
        Dwx = wx
        Dwnu = T.tensordot(Gammanu,wx,(2,0))+wnu

        return T.concatenate((Dwx,Dwnu.flatten()))
    def from_D(u,Dw):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dwx = Dw[0:d]
        Dwnu = Dw[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = T.tensordot(M.Gamma_g(x),nu,(2,0)).dimshuffle(0,2,1)
        wx = Dwx
        wnu = -T.tensordot(Gammanu,Dwx,(2,0))+Dwnu

        return T.concatenate((wx,wnu.flatten())) 
        # corresponding dual space shifts
    def to_Dstar(u,p):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        px = p[0:d]
        pnu = p[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = T.tensordot(M.Gamma_g(x),nu,(2,0)).dimshuffle(0,2,1)
        Dpx = px-T.tensordot(Gammanu,pnu,((0,1),(0,1)))
        Dpnu = pnu

        return T.concatenate((Dpx,Dpnu.flatten()))
    def from_Dstar(u,Dp):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dpx = Dp[0:d]
        Dpnu = Dp[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = T.tensordot(M.Gamma_g(x),nu,(2,0)).dimshuffle(0,2,1)
        px = Dpx+T.tensordot(Gammanu,Dpnu,((0,1),(0,1)))
        pnu = Dpnu

        return T.concatenate((px,pnu.flatten()))
    M.to_D = to_D
    M.to_Df = M.coords_function(M.to_D,w)      
    M.from_D = from_D
    M.from_Df = M.coords_function(M.from_D,w)        
    M.to_Dstar = to_Dstar
    M.to_Dstarf = M.coords_function(M.to_Dstar,p)      
    M.from_Dstar = from_Dstar
    M.from_Dstarf = M.coords_function(M.from_Dstar,p)        
    
    ##### Horizontal vector fields:
    def Horizontal(u):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
    
        # Contribution from the coordinate basis for x: 
        dx = nu
        # Contribution from the basis for Xa:
        dnu = -T.tensordot(nu, T.tensordot(nu, M.Gamma_g(x),(0,2)),(0,2))

        dnuv = dnu.reshape((nu.shape[1],dnu.shape[1]*dnu.shape[2]))

        return T.concatenate([dx,dnuv.T],axis = 0)
    M.Horizontal = Horizontal
    M.Horizontalf = M.coords_function(M.Horizontal)
    
    ##### Cometric and Hamiltonian
    def g_FMsharp(u):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,d))#.reshape((d,M.m))
        GamX = T.tensordot(M.Gamma_g(x), nu, axes = [2,0]).dimshuffle(0,2,1)
    
        delta = T.eye(nu.shape[0],nu.shape[1])
        Winv = T.tensordot(nu,  nu,  axes = [1,1]) + lambdag0*M.g(x)
    
        gij = Winv
        gijb = -T.tensordot(Winv, GamX, axes = [1,2])
        giaj = -T.tensordot(GamX, Winv, axes = [2,0])
        giajb = T.tensordot(T.tensordot(GamX, Winv, axes = [2,0]), 
                            GamX, axes = [2,2])

        return gij,gijb,giaj,giajb
        
    def Dg_FMDstar(u,Dp):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dpx = Dp[0:d]
        Dpnu = Dp[d:].reshape((d,-1))
        
        Winv = T.tensordot(nu, nu, axes = [1,1])        
        DgDpx = T.tensordot(Winv,Dpx,(1,0))
        
        return T.concatenate((DgDpx,T.zeros_like(Dpnu).flatten()))
    M.Dg_FMDstar = Dg_FMDstar
    M.Dg_FMDstarf = M.coords_function(M.Dg_FMDstar,p)    
    
    def H_FM(u,p):
        Dp = M.to_Dstar(u,p)
        Dgp = Dg_FMDstar(u,Dp)
        
        return 0.5*T.dot(Dp,Dgp)
#     def H_FM(u,p):
#         x = (u[0][0:d],u[1])
#         nu = u[0][d:].reshape((d,-1))
#         px = p[0:d]
#         pnu = p[d:].reshape((d,-1))
        
#         GamX = T.tensordot(M.Gamma_g(x), nu, 
#                            axes = [2,0]).dimshuffle(0,2,1)
    
#         Winv = T.tensordot(nu, nu, axes = [1,1])
    
#         gij = Winv
#         gijb = -T.tensordot(Winv, GamX, axes = [1,2])
#         giaj = -T.tensordot(GamX, Winv, axes = [2,0])
#         giajb = T.tensordot(T.tensordot(GamX, Winv, axes = [2,0]), 
#                             GamX, axes = [2,2])
    
#         pxgpx = T.dot(T.tensordot(px, gij, axes = [0,0]), px)
#         pxgpnu = T.tensordot(T.tensordot(px, gijb, axes = [0,0]), 
#                              pnu, axes = [[0,1],[0,1]])
#         pnugpx = T.tensordot(T.tensordot(px, giaj, axes = [0,2]), 
#                              pnu, axes = [[0,1],[0,1]])
#         pnugpnu = T.tensordot(T.tensordot(giajb, pnu, axes = [[2,3],[0,1]]), 
#                               pnu, axes = [[0,1],[0,1]])
    
#         return 0.5*(pxgpx + pxgpnu + pnugpx + pnugpnu)
    M.H_FM = H_FM
    M.H_FMf = M.coords_function(M.H_FM,p)


    ##### Evolution equations:
    dq = lambda q,p: T.grad(M.H_FM(q,p),p)
    dp = lambda q,p: -T.grad(M.H_FM(q,p),q[0])

    def ode_Hamiltonian_FM(t,x,chart): # Evolution equations at (p,q).
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        return T.stack((dqt,dpt))

    def chart_update_Hamiltonian_FM(t,up,chart):
        if do_chart_update is None:
            return (t,up,chart)
        
        u = up[0]
        p = up[1]
        Dp = M.to_Dstar((u,chart),p)

        x = (u[0:d],chart)
        nu = u[d:].reshape((d,-1))
        Dpx = Dp[0:d]
        Dpnu = Dp[d:].reshape((d,-1))

        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
        new_nu = M.update_vector(x,new_x,new_chart,nu)
        new_u = T.concatenate((new_x,new_nu.flatten()))
        new_Dpx = M.update_covector(x,new_x,new_chart,Dpx)
        new_Dpnu = M.update_covector(x,new_x,new_chart,Dpnu)
        new_Dp = T.concatenate((new_Dpx,new_Dpnu.flatten()))
        
        return theano.ifelse.ifelse(do_chart_update(x),
                (t,up,chart),(t,T.stack((theano.gradient.disconnected_grad(new_u),
                                         theano.gradient.disconnected_grad(M.from_Dstar((new_u,new_chart),new_Dp)))),
                              new_chart)
            )
    M.chart_update_Hamiltonian_FM = chart_update_Hamiltonian_FM
    M.chart_update_Hamiltonian_FMf = M.coords_function(lambda u,p: M.chart_update_Hamiltonian_FM(0.,(u[0],p),u[1])[1:],p)

    M.Hamiltonian_dynamics_FM = lambda q,p: integrate(ode_Hamiltonian_FM,chart_update_Hamiltonian_FM,T.stack((q[0],p)),q[1])
    M.Hamiltonian_dynamics_FMf = M.coords_function(M.Hamiltonian_dynamics_FM,p)

    ## Geodesic
    def Exp_Hamiltonian_FM(u,p):
        curve = M.Hamiltonian_dynamics_FM(u,p)
        u = curve[1][-1,0]
        chart = curve[2][-1]
        return(u,chart)
    M.Exp_Hamiltonian_FM = Exp_Hamiltonian_FM
    M.Exp_Hamiltonian_FMf = M.coords_function(M.Exp_Hamiltonian_FM,p)
    def Exp_Hamiltonian_FMt(u,p):
        curve = M.Hamiltonian_dynamics_FM(u,p)
        u = curve[1][:,0]
        chart = curve[2]
        return(u,chart)
    M.Exp_Hamiltonian_FMt = Exp_Hamiltonian_FMt
    M.Exp_Hamiltonian_FMtf = M.coords_function(M.Exp_Hamiltonian_FMt,p)

    # Most probable path for the driving semi-martingale
    M.loss = lambda u,x,p: 1./d*T.sum((M.Exp_Hamiltonian_FM(u,p)[0][0:d] - x[0:d])**2)
    M.lossf = M.coords_function(M.loss,x,p)

    def Log_FM(u,x):
        def fopts(p):
            y = M.lossf(u,x,p)
            return y

        res = minimize(fopts, np.zeros(u.shape), 
                       method='CG', jac=False, options={'disp': False, 
                                                        'maxiter': 50})
        return res.x
    M.Log_FM = Log_FM

    