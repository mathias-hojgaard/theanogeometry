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
#try:
#    from src.manifold import *
#except NameError:
#    pass
#try:
#    from src.group import *
#except NameError:
#    pass

#######################################################################
# various useful functions                                            #
#######################################################################

def scalar(x):
    """ return scalar of the default Theano type """
    return np.float32(x) if theano.config.floatX == 'float32' else np.float64(x)
def tensor(x):
    """ return tensor of the default Theano type """
    return np.array(x).astype(theano.config.floatX)
def constant(c):
    """ return Theano constant with value of parameter c """
    if isinstance(c,int):
        return T.constant(c)
    if isinstance(c,float):
        return T.constant(scalar(c))
    if isinstance(c,T.TensorConstant):
        return c
    raise ValueError()

# get function derivatives and compiled versions
def get_df(f,x,thetas,extra_params):
    y = f(x,*extra_params)
    dfx = T.grad(y,x)
    dfthetas = tuple(T.grad(y, theta) for theta in thetas)
    
    df = (y,dfx)+dfthetas
    dff = theano.function([x]+list(extra_params), df)
    return (df,dff)

# numeric optimizer
def get_minimizer(f,method=None,options=None):
    x = T.vector()
    ff = theano.function([x],f(x))
    gradff = theano.function([x],T.grad(f(x),x))

    def fopt(x):
        return (ff(x),gradff(x))

    return (lambda x0: minimize(fopt, x0, method=method, jac=True, options=None))

# Integrator (non-stochastic)
def integrator(ode_f,method=default_method,use_charts=False,chart_update=None):
    if chart_update == None: # no chart update
        chart_update = lambda *args: args

    # euler:
    def euler(*y):
        if not use_charts:
            t = y[-2]
            x = y[-1]
            return (t+dt,x+dt*ode_f(*y))
        else:
            t = y[-3]
            x = y[-2]
            chart = y[-1]
            return chart_update(t+dt,x+dt*ode_f(*y),chart,*y[0:-3])

    # Runge-kutta:
    def rk4(*y):
        if not use_charts:
            t = y[-2]
            x = y[-1]
            k1 = ode_f(y[0:-2],t,x)
            k2 = ode_f(y[0:-2],t+dt/2,x + dt/2*k1)
            k3 = ode_f(y[0:-2],t+dt/2,x + dt/2*k2)
            k4 = ode_f(y[0:-2],t,x + dt*k3)
            return (t+dt,x + dt/6*(k1 + 2*k2 + 2*k3 + k4))
        else:
            t = y[-3]
            x = y[-2]
            chart = y[-1]
            k1 = ode_f(y[0:-2],t,x)
            k2 = ode_f(y[0:-2],t+dt/2,x + dt/2*k1)
            k3 = ode_f(y[0:-2],t+dt/2,x + dt/2*k2)
            k4 = ode_f(y[0:-2],t,x + dt*k3)
            return chart_update(t+dt,x + dt/6*(k1 + 2*k2 + 2*k3 + k4),chart,*y[0:-3])

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

# return symbolic path given ode and integrator
def integrate(ode,chart_update,x,chart,*y):
    if chart is None:
        (cout, updates) = theano.scan(fn=integrator(ode),
                outputs_info=[constant(0.),x],
                sequences=[*y],
                n_steps=n_steps)
    else:
        (cout, updates) = theano.scan(fn=integrator(ode,use_charts=True,chart_update=chart_update),
                outputs_info=[constant(0.),x,chart],
                sequences=[*y],
                n_steps=n_steps)
    return cout

# sde functions should return (det,sto,Sigma) where
# det is determinisitc part, sto is stochastic part,
# and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)

# standard noise realisations
srng = RandomStreams()#seed=42)
dWt = T.matrix() # n_steps x d, d usually manifold dimension
dWs = lambda d: srng.normal((n_steps,d), std=T.sqrt(dt))
d = T.scalar(dtype='int64')
dWsf = theano.function([d],dWs(d))
del d

def integrate_sde(sde,integrator,chart_update,x,chart,dWt,*ys):
    if chart is None:
        (cout, updates) = theano.scan(fn=integrator(sde),
                outputs_info=[constant(0.),x,*ys],
                sequences=[dWt],
                n_steps=n_steps)
    else:
        (cout, updates) = theano.scan(fn=integrator(sde,use_charts=True,chart_update=chart_update),
                outputs_info=[constant(0.),x,chart,*ys],
                sequences=[dWt],
                n_steps=n_steps)
    return cout

def integrator_stratonovich(sde_f,use_charts=False,chart_update=None):
    if chart_update == None: # no chart update
        chart_update = lambda *args: args

    def euler_heun(*y):
        if not use_charts:
            (dW,t,x,*ys) = y
            (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
            tx = x + stox
            ys_new = ()
            for (y,dy) in zip(ys,dys):
                ys_new = ys_new + (y+dt*dy,)
            return (t+dt,x + dt*detx + 0.5*(stox + sde_f(dW,t+dt,tx,*ys)[1]), *ys_new)
        else:
            (dW,t,x,chart,*ys) = y
            (detx, stox, X, *dys) = sde_f(dW,t,x,chart,*ys)
            tx = x + stox
            ys_new = ()
            for (y,dy) in zip(ys,dys):
                ys_new = ys_new + (y+dt*dy,)
            return chart_update(t+dt,x + dt*detx + 0.5*(stox + sde_f(dW,t+dt,tx,chart,*ys)[1]), chart, *ys_new)

    return euler_heun

def integrator_ito(sde_f,use_charts=False,chart_update=None):
    if chart_update == None: # no chart update
        chart_update = lambda *args: args

    def euler(*y):
        if not use_charts:
            (dW,t,x,*ys) = y
            (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
            ys_new = ()
            for (y,dy) in zip(ys,dys):
                ys_new = ys_new + (y+dt*dy,)
            return (t+dt,x + dt*detx + stox, *ys_new)
        else:
            (dW,t,x,chart,*ys) = y
            (detx, stox, X, *dys) = sde_f(dW,t,x,chart,*ys)
            ys_new = ()
            for (y,dy) in zip(ys,dys):
                ys_new = ys_new + (y+dt*dy,)
            return chart_update(t+dt,x + dt*detx + stox, chart, *ys_new)

    return euler

## Gram-Schmidt:
def GramSchmidt_f(innerProd):
    def GS(q,Frame):

        if len(Frame.shape) == 1:
            gS = Frame/np.sqrt(innerProd(q,Frame,Frame))
        else:
            gS = np.zeros_like(Frame)
            for j in range(0,Frame.shape[1]):
                gS[:,j] = Frame[:,j]
                for i in range(0,j):
                    foo = innerProd(q,Frame[:,j],gS[:,i])/ innerProd(q,gS[:,i],gS[:,i])
                    gS[:,j] = gS[:,j] - foo*gS[:,i]

                gS[:,j] = gS[:,j]/np.sqrt(innerProd(q,gS[:,j],gS[:,j]))

        return gS

    return GS

def cross(a, b):
    return T.as_tensor([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]) 


