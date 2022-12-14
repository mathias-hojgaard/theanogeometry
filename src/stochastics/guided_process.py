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

from src.utils import *
import src.linalg as linalg

#######################################################################
# guided processes, Delyon/Hu 2006                                    #
#######################################################################

# hit target v at time t=Tend
def get_sde_guided(M, sde_f, phi, sqrtCov, A=None, method='DelyonHu', integration='ito', use_charts=False, chart_update=None, v_chart_update=None):
    assert (integration == 'ito' or integration == 'stratonovich')
    assert (method == 'DelyonHu')  # more general schemes not implemented
    
    def chart_update_guided(t, x, chart, log_likelihood, log_varphi, h, v, *ys):
        if chart_update is None:
            return (t, x, chart, log_likelihood, log_varphi, h, v, *ys)

        (t_new, x_new, chart_new, *ys_new) = chart_update(t,x,chart,*ys)
        v_new = v if v_chart_update is None else M.update_coords((v,chart),chart_new)[0]
        return (t_new, x_new, chart_new, log_likelihood, log_varphi, h, v_new, *ys_new)

    def sde_guided(dW, t, x, chart, log_likelihood, log_varphi, h, v, *ys):
        if not use_charts:
            (det, sto, X, *dys_sde) = sde_f(dW, t, x, *ys)
        else:
            (det, sto, X, *dys_sde) = sde_f(dW, t, x, chart, *ys)
            
        xchart = x if not use_charts else (x,chart)

        h = theano.ifelse.ifelse(T.lt(t, Tend - dt / 2),
                                 phi(xchart, v) / (Tend - t),
                                 T.zeros_like(phi(xchart, v))
                                 )
        sto = theano.ifelse.ifelse(T.lt(t, Tend - 3 * dt / 2),  # for Ito as well?
                                   sto,
                                   T.zeros_like(sto)
                                   )

        ### likelihood
        dW_guided = (1 - .5 * dt / (1 - t)) * dW + dt * h  # for Ito as well?
        sqrtCovx = sqrtCov(xchart)
        Cov = dt * T.tensordot(sqrtCovx, sqrtCovx, (1, 1))
        Pres = T.nlinalg.MatrixInverse()(Cov)
        residual = T.tensordot(dW_guided, T.tensordot(Pres, dW_guided, (1, 0)), (0, 0))
        log_likelihood = .5 * (-dW.shape[0] * T.log(2 * np.pi) + linalg.LogAbsDet()(Pres) - residual)

        ## correction factor
        ytilde = T.tensordot(X, h * (Tend - t), 1)
        tp1 = t + dt
        if integration == 'ito':
            xtp1 = x + dt * (det + T.tensordot(X, h, 1)) + sto
        elif integration == 'stratonovich':
            tx = x + sto
            xtp1 = x + dt * det + 0.5 * (sto + sde_f(dW, tp1, tx, *ys)[1])
        xtp1chart = xtp1 if not use_charts else (xtp1,chart)
        if not use_charts:
            Xtp1 = sde_f(dW, tp1, xtp1, *ys)[2]
        else:
            Xtp1 = sde_f(dW, tp1, xtp1, chart, *ys)[2]
        ytildetp1 = T.tensordot(Xtp1, phi(xtp1chart, v), 1)

        # set default A if not specified
        Af = A if A is not None else lambda x, v, w: T.tensordot(v, T.tensordot(T.nlinalg.MatrixInverse()(T.tensordot(X, X, (1, 1))), w, 1), 1)

        #     add t1 term for general phi
        #     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = theano.ifelse.ifelse(T.lt(t, Tend - 3 * dt / 2),
                                  -Af(xchart, ytilde, det * dt) / (Tend - t),
                                  # check det term for Stratonovich (correction likely missing)
                                  constant(0.))
        t34 = theano.ifelse.ifelse(T.lt(tp1, Tend - 3 * dt / 2),
                                   -(Af(xtp1chart, ytildetp1, ytildetp1) - Af(xchart, ytildetp1, ytildetp1)) / (
                                   2 * (Tend - tp1 + dt * T.gt(tp1, Tend - dt / 2))),
                                   # last term in divison is to avoid NaN with non-lazy Theano conditional evaluation
                                   constant(0.))
        log_varphi = t2 + t34

        return (det + T.tensordot(X, h, 1), sto, X, log_likelihood, log_varphi/dt, dW_guided/dt, T.zeros_like(v), *dys_sde)

    if not use_charts:
        return lambda dW, t, x, log_likelihood, log_varphi, h, v, *ys: sde_guided(dW, t, x, None, log_likelihood, log_varphi, h, v, *ys)
    else:
        return (sde_guided, chart_update_guided)

def get_guided_likelihood(M, sde_f, phi, sqrtCov, A=None, method='DelyonHu', integration='ito', use_charts=False, chart_update=None):
    v = M.sym_element()
    if not use_charts:
        sde_guided = get_sde_guided(M, sde_f, phi, sqrtCov, A, method, integration)
        guided = lambda q, v, dWt: integrate_sde(sde_guided,
                                                 integrator_ito if method == 'ito' else integrator_stratonovich,
                                                 None,
                                                 q, None, dWt, constant(0.), constant(0.), T.zeros_like(dWt[0]), v)
        guidedf = M.function(guided,v,dWt)
    else:
        (sde_guided,chart_update_guided) = get_sde_guided(M, sde_f, phi, sqrtCov, A, method, integration, use_charts=True, chart_update=chart_update)
        guided = lambda q, v, dWt: integrate_sde(sde_guided,
                                                 integrator_ito if method == 'ito' else integrator_stratonovich,
                                                 chart_update_guided,
                                                 q[0], q[1], dWt, constant(0.), constant(0.), T.zeros_like(dWt[0]), v)
        guidedf = M.coords_function(guided,v,dWt)

    return (guided, guidedf)

import src.linalg as linalg

def bridge_sampling(lg,bridge_sdef,dWsf,options,pars):
    """ sample samples_per_obs bridges """
    (v,seed) = pars
    if seed:
        srng.seed(seed)
    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
    log_varphis = np.zeros((options['samples_per_obs'],))
    log_likelihoods = np.zeros((options['samples_per_obs'],))
    for i in range(options['samples_per_obs']):
        (ts,gs,log_likelihood,log_varphi) = bridge_sdef(lg,v,dWsf())[:4]
        bridges[i] = gs
        log_varphis[i] = log_varphi[-1]
        log_likelihoods[i] = log_likelihood[-1]
        try:
            v = options['update_vf'](v) # update v, e.g. simulate in fiber
        except KeyError:
            pass
    return (bridges,log_varphis,log_likelihoods,v)

# helper for log-transition density
def p_T_log_p_T(g, v, dWs, bridge_sde, phi, options, F=None, sde=None, use_charts=False, chain_sampler=None, init_chain=None):
    """ Monte Carlo approximation of log transition density from guided process """
    if use_charts:
        chart = g[1]
    
    # sample noise
    (cout, updates) = theano.scan(fn=lambda x: dWs,
                                  outputs_info=[T.zeros_like(dWs)],
                                  n_steps=options['samples_per_obs'])
    dWsi = cout
    
    # map v to M
    if F is not None:
        v = F(v if not use_charts else (v,chart))

    if not 'update_v' in options:
        # v constant throughout sampling
        print("transition density with v constant")
        
        # bridges
        Cgv = T.sum(phi(g, v) ** 2)
        def bridge_logvarphis(dWs, log_varphi, chain):
            if chain_sampler is None:
                w = dWs
            else:
                (accept,new_w) = chain_sampler(chain)
                w = T.switch(accept,new_w,w)
            if not use_charts:
                (ts, gs, log_likelihood, log_varphi) = bridge_sde(g, v, theano.gradient.disconnected_grad(w))[:4] # we don't take gradients of the sampling scheme
            else:
                (ts, gs, charts, log_likelihood, log_varphi) = bridge_sde(g, v, theano.gradient.disconnected_grad(w))[:5] # we don't take gradients of the sampling scheme
            return (log_varphi[-1], w)

        (cout, updates) = theano.scan(fn=bridge_logvarphis,
                                      outputs_info=[constant(0.),init_chain if init_chain is not None else T.zeros_like(dWs)],
                                      sequences=[dWsi])
        log_varphi = T.log(T.mean(T.exp(cout[0])))
        log_p_T = -.5 * g[0].shape[0] * T.log(2. * np.pi * Tend) - Cgv / (2. * Tend) + log_varphi
        p_T = T.exp(log_p_T)
    else:
        # update v during sampling, e.g. for fiber densities
        assert(chain_sampler is None)
        print("transition density with v updates")

        # bridges
        def bridge_p_T(dWs, lp_T, lv):
            Cgv = T.sum(phi(g, lv) ** 2)
            (ts, gs, log_likelihood, log_varphi) = bridge_sde(g, lv, dWs)[:4]
            lp_T =  T.power(2.*np.pi*Tend,-.5*g[0].shape[0])*T.exp(-Cgv/(2.*Tend))*T.exp(log_varphi[-1])
            lv = options['update_v'](lv)                        
            return (lp_T, lv)

        (cout, updates) = theano.scan(fn=bridge_p_T,
                                      outputs_info=[constant(0.), v],
                                      sequences=[dWsi])
        p_T = T.mean(cout[:][0])
        log_p_T = T.log(p_T)
        v = cout[-1][1]
    
    if chain_sampler is None:
        return (p_T,log_p_T,v)
    else:
        return (p_T,log_p_T,v,w)

# densities wrt. the Riemannian volume form
def p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[0]
def log_p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[1]

def dp_T(thetas,*args,**kwargs):
    """ Monte Carlo approximation of transition density gradient """
    lp_T = p_T(*args,**kwargs)
    return (lp_T,)+tuple(T.grad(lp_T,theta) for theta in thetas)

def dlog_p_T(thetas,*args,**kwargs):
    """ Monte Carlo approximation of log transition density gradient """
    llog_p_T = log_p_T(*args,**kwargs)
    return (llog_p_T,)+tuple(T.grad(llog_p_T,theta) for theta in thetas)
