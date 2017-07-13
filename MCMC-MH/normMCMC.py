
#  posterior distribution for Poisson model
import scipy.stats as _ss
import os

twpi  = 2*_N.pi
ITERS = 40000

us      = _N.empty(ITERS)
ts      = _N.empty(ITERS)   
sd2s    = _N.empty(ITERS)
lls     = _N.empty(ITERS)
llklhdrs= _N.empty(ITERS)
lprps   = _N.empty(ITERS)

N     = 40
ugt  = 1.8
s2gt = 0.5**2

##  filename poisson   #poidat,N= ,mn= .dat
datfn = "norm,N=%(N)d,mn=%(mn).1f,sd=%(sd).1f" % {"N" : N, "mn" : ugt, "sd" : _N.sqrt(s2gt)}

if not os.access(datfn, os.F_OK):
    dat  = ugt + _N.sqrt(s2gt)*_N.random.randn(N)
    _N.savetxt(datfn, dat, fmt="%.5f")
else:
    dat  = _N.loadtxt(datfn)

#  proposal params
#  p(u_1) = u_0 + N(0, delta^2)
delta    = 0.05
#  p(sd2_1 | u) = Exponential(mean = u_0^2)
c        = 1.
revpr  = False

datfn = "norm,N=%(N)d,mn=%(mn).1f,sd=%(sd).1f" % {"N" : N, "mn" : ugt, "sd" : _N.sqrt(s2gt)}

mpr    = -1 if revpr else 1

sd2_0 = 0.3
u_0   = 1.5

#  sd

ll0 = -0.5*N*_N.log(twpi*sd2_0) - 0.5*_N.sum((dat-u_0)*(dat-u_0))/sd2_0

for it in xrange(ITERS):
    #  k    p(x) = k e^{-kx}.   <x> = 1/k.  Make k(lm)

    us[it] = u_0
    u_1    = u_0 + delta*_N.random.randn()
    sd2_1 =  _N.random.exponential(c*u_1*u_1)  #  mean is u_1*u_1

    sd2s[it] = sd2_0
    lls[it]  = ll0
    #print "%(u).2f  %(sd).3f" % {"u" : u_1, "sd" : sd2_1}

    ll1 = -0.5*N*_N.log(twpi*sd2_1) - 0.5*_N.sum((dat-u_1)*(dat-u_1))/sd2_1

    llklhdrs[it] = ll1-ll0

    lprps[it] = mpr*(_N.log(c*u_1*u_1) - _N.log(c*u_0*u_0) - sd2_0/(c*u_0*u_0) + sd2_1 / (c*u_1*u_1))

    lalp       = ll1 - ll0 + lprps[it]

    accpt = False
    if lalp > 0:
        accpt = True
    elif _N.random.rand() < _N.exp(lalp):
        accpt = True
    if accpt:
        u_0 = u_1
        sd2_0 = sd2_1
        ll0 = ll1


#  show iterations us, sd2s
#  histogram of these as well
fig = _plt.figure(figsize=(9, 8))
ax  = fig.add_subplot(2, 2, 1)
_plt.plot(us)
ax  = fig.add_subplot(2, 2, 2)
_plt.plot(sd2s)
ax  = fig.add_subplot(2, 2, 3)
_plt.hist(us[1000:], bins=_N.linspace(_N.min(us[1000:]), _N.max(us[1000:]), 100))
ax  = fig.add_subplot(2, 2, 4)
_plt.hist(sd2s[1000:], bins=_N.linspace(_N.min(sd2s[1000:]), _N.max(sd2s[1000:]), 100))
