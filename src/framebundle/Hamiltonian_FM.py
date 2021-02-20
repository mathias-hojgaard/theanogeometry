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

############################################################################################
# Hamiltionian dynamics on FM from sub-Riemannian structure <v,w>_FM=<u^-1(v),u^-1(w)>_R^2 #
############################################################################################
def initialize(M,use_charts=True):
    
    d  = M.dim

    x = M.sym_element()
    x1 = M.sym_element()
    v = M.sym_vector()
    nu = M.sym_frame()

    u = M.sym_FM_element()
    q = M.sym_FM_element()
    w = M.sym_FM_vector()    
    p = M.sym_FM_covector()    
    
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
        Dgp = M.Dg_FMDstar(u,Dp)
        
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
        if M.do_chart_update is None:
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
        
        return theano.ifelse.ifelse(M.do_chart_update(x),
                (t,up,chart),(t,T.stack((theano.gradient.disconnected_grad(new_u),
                                         theano.gradient.disconnected_grad(M.from_Dstar((new_u,new_chart),new_Dp)))),
                              new_chart)
            )
    M.chart_update_Hamiltonian_FM = chart_update_Hamiltonian_FM
    M.chart_update_Hamiltonian_FMf = M.coords_function(lambda u,p: M.chart_update_Hamiltonian_FM(0.,(u[0],p),u[1])[1],p)

    M.Hamiltonian_dynamics_FM = lambda q,p: integrate(ode_Hamiltonian_FM,chart_update_Hamiltonian_FM,T.stack((q[0],p)),q[1])
    M.Hamiltonian_dynamics_FMf = M.coords_function(M.Hamiltonian_dynamics_FM,p)

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

        res = minimize(fopts, np.zeros(u[0].shape), 
                       method='CG', jac=False, options={'disp': False, 
                                                        'maxiter': 50})
        return res.x
    M.Log_FM = Log_FM