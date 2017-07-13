
#  posterior distribution for Poisson model
import scipy.stats as _ss
import os

logfact = depickle("/Users/arai/usb/nctc/Workspace/PP-AR/pyscripts/logfact.dump")

_EXPO = 1
_NORM = 2

ITERS = 10000


lmds    = _N.empty(ITERS)
lls     = _N.empty(ITERS)
llklhdrs= _N.empty(ITERS)
lprps   = _N.empty(ITERS)

N     = 40
lamgt = 5

##  filename poisson   #poidat,N= ,mn= .dat
datfn = "poiss,N=%(N)d,mn=%(mn).1f" % {"N" : N, "mn" : lamgt}
if not os.access(datfn, os.F_OK):
    cnts  = _ss.poisson.rvs(lamgt, size=N)
    _N.savetxt(datfn, cnts, fmt="%d")
else:
    cnts  = _N.loadtxt(datfn, dtype=_N.int)


lm0   = 1

sumcnts= _N.sum(cnts)
sd    = 0.1

twpi  = 2*_N.pi

prop   = _EXPO
revpr  = False
mpr    = -1 if revpr else 1

sumlogfactcnts = _N.sum(logfact[cnts])
ll0 = sumcnts * _N.log(lm0) - N*lm0 - sumlogfactcnts   # current

for it in xrange(ITERS):
    lmds[it] = lm0
    lls[it]  = ll0

    #  k    p(x) = k e^{-kx}.   <x> = 1/k.  Make k(lm)
    if prop == _EXPO:
        k1    = lm0
        lm1   = _N.random.exponential(k1)   # mean of lm1 is k1=lm0.  
        k0    = lm1
        lpr1 = -_N.log(k1) - lm1/k1
        lpr0 = -_N.log(k0) - lm0/k0
    elif prop == _NORM:
        sd1 = (lm0/2.)*(lm0/2.)*sd
        lm1 = lm0 + sd1*_N.random.randn()   # proposed lambda
        sd0 = (lm1/2.)*(lm1/2.)*sd
        #forward
        lpr1  = -0.5*_N.log(twpi*sd1*sd1) - 0.5*(lm1-lm0)*(lm1-lm0)/(sd1*sd1) 
        #backward
        lpr0  = -0.5*_N.log(twpi*sd0*sd0) - 0.5*(lm0-lm1)*(lm0-lm1)/(sd0*sd0)

    ll1 = sumcnts * _N.log(lm1) - N*lm1 - sumlogfactcnts   # proposed

    lalp = ll1-ll0 + mpr*(lpr0 - lpr1)
    llklhdrs[it] = ll1-ll0
    lprps[it] = lpr0-lpr1

    accpt = False
    if lalp > 0:
        accpt = True
    elif _N.random.rand() < _N.exp(lalp):
        accpt = True

    if accpt:
        lm0 = lm1
        ll0 = ll1

