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

# from src.statistics.Frechet_mean import *
def Frechet_mean(Logf, y, x0=None, options=None):
    y = list(y) # make sure y is subscriptable

    if x0 is None:
        #x0 = np.random.normal(size=M.dim.eval())
        x0 = y[np.random.randint(len(y))]
    chart = x0[1]
        
    steps = []
    steps.append(x0[0])

    def fopts(x):
        N = len(y)
        sol = mpu.pool.imap(lambda pars: (Logf((x,chart),y[pars[0]])[0],),mpu.inputArgs(range(N)))
        res = list(sol)
        Logs = mpu.getRes(res,0)
#         Logs = np.zeros((N, x.shape[0]))
#         for i in range(N):
#             Logs[i] = Logf((x,chart), y[i])[0]

        res = (1. / N) * np.sum(np.square(Logs))
        grad = -(2. / N) * np.sum(Logs, 0)

        return (res, grad)

    def save_step(k):
        steps.append(k)

    try:
        mpu.openPool()
        res = minimize(fopts, x0[0], method='BFGS', jac=True, options=options, callback=save_step)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()

    return ((res.x,x0[1]), res.fun, np.array(steps))