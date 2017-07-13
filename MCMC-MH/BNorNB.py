import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt
import time as _tm
import scipy.stats as _ss
import os

#  connection between 
logfact= None

ints = _N.arange(20000)
ln2pi= 1.8378770664093453

def _init(lf):
    global logfact 
    logfact = lf

def Llklhds(typ, ks, rn, p):
    global logfact
    N = len(ks)
    if typ == _cd.__BNML__:
        return N*logfact[rn]-_N.sum(logfact[ks]+logfact[rn-ks]-ks*_N.log(p) - (rn-ks)*_N.log(1 - p))
    else:
        return _N.sum(logfact[ks+rn-1]-logfact[ks]  + ks*_N.log(p) + rn*_N.log(1 - p))-N*logfact[rn-1]


def BNorNBonly(GibbsIter, iters, w, j, u0, rn0, dist, cts, rns, us, dty, xn, jxs, jmp, lls, accpts, llklhd0, llklhd1, lrats, lpprs0, lpprs1, stdu=0.03, propratm=0):
    global ints, logfact
    #  if accptd too small, increase stdu and try again
    #  if accptd is 0, the sampled params returned for conditional posterior
    #  are not representative of the conditional posterior
    stdu2= stdu**2
    istdu2= 1./ stdu2
    #stdp2 = 0.25*0.25
    #stdp = 0.002
    stdp = 0.05
    istdp= 1./stdp
    istdp2= istdp*istdp

    Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    if Mk == 0:
        return u0, rn0, dist   # no data assigned to this 
    if dist == _cd.__BNML__:
        rnmin= _N.max(cts)+1 if len(cts) > 0 else 0   #  if n too small, can't generate data
    else:
        rnmin = 1

    rdis= _N.random.rand(iters)  #
    rdns = _N.random.randn(iters)
    ran_accpt  = _N.random.rand(iters)

    p0  = 1/(1 + _N.exp(-u0))

    ll0 = Llklhds(dist, cts, rn0, p0)
    #  the poisson distribution needs to be truncated
    prop_ps = _N.empty(iters)
    prop_rns = _N.empty(iters, dtype=_N.int)

    zr2nmin  = _N.arange(rnmin) # rnmin is a valid value for n
    for it in xrange(iters):
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted
        lls[it] = ll0

        if it % 1000 == 0:
            print it

        ##  propose an rn1 
        m2          = 1./rn0 + 0.2      # rn0 is the mean for proposal for rn1
        p_prp_rn1        = 1 - 1./(rn0*m2)  # param p for proposal for rn1
        r_prp_rn1        = rn0 / (rn0*m2-1) # param r for proposal for rn1
        
        bGood = False   #  rejection sampling of rn1
        while not bGood:
            rn1 = _N.random.negative_binomial(r_prp_rn1, 1-p_prp_rn1)
            if rn1 >= rnmin:
                bGood = True
        prop_rns[it] = rn1

        #########  log proposal density for rn1
        ir_prp_rn1 = int(r_prp_rn1)
        ltrms = logfact[zr2nmin+ir_prp_rn1-1]  - logfact[ir_prp_rn1-1] - logfact[zr2nmin] + ir_prp_rn1*_N.log(1-p_prp_rn1) + zr2nmin*_N.log(p_prp_rn1)
        lCnb1        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated pmf
        lpmf1       = logfact[rn1+r_prp_rn1-1]  - logfact[r_prp_rn1-1] - logfact[rn1] + r_prp_rn1*_N.log(1-p_prp_rn1) + rn1*_N.log(p_prp_rn1) - lCnb1

        #########  log proposal density for rn0
        ##  rn1
        m2          = 1./rn1 + 0.2      # rn0 is the mean for proposal for rn1
        p_prp_rn0        = 1 - 1./(rn1*m2)  # param p for proposal for rn1
        r_prp_rn0        = rn1 / (rn1*m2-1.) # param r for proposal for rn1
        ir_prp_rn0 = int(r_prp_rn0)

        ltrms = logfact[zr2nmin+ir_prp_rn0-1]  - logfact[ir_prp_rn0-1] - logfact[zr2nmin] + ir_prp_rn0*_N.log(1-p_prp_rn0) + zr2nmin*_N.log(p_prp_rn0)
        lCnb0        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated 
        
        lpmf0       = logfact[rn0+r_prp_rn0-1]  - logfact[r_prp_rn0-1] - logfact[rn0] + r_prp_rn0*_N.log(1-p_prp_rn0) + rn0*_N.log(p_prp_rn0) - lCnb0

        ###################################################
        #  propose p1
        #  sample using p from [0, 0.75]  mean 0.25, sqrt(variance) = 0.25
        #  p1 x rn1 = p0 x rn0      --> p1 = p0 x rn0/rn1   BINOMIAL
        #  
        mu_p1  = (p0 * rn0)/rn1 if dist == _cd.__BNML__ else (rn0*p0)/(rn0*p0 + rn1*(1-p0))
        stdp1 = (0.5-_N.abs(mu_p1-0.5))*0.05 + 0.0001
        istdp1= 1./stdp1
        limL    = -mu_p1*istdp1     #  0-mu_p1
        limR    = (1-mu_p1)*istdp1  #  1-mu_p1
        limRmL  = limR-limL
        p1      = _ss.truncnorm.rvs(limL, limR)  #  
        p1      = (p1-limL)/limRmL
        prop_ps[it] = p1

        nC1    = _ss.norm.cdf((1-mu_p1)*istdp1) - _ss.norm.cdf(-mu_p1*istdp1)
        lnC1   = _N.log(nC1)
        u1    = -_N.log(1./p1 -1)
        mu_p0  = (p1 * rn1)/rn0 if dist == _cd.__BNML__ else (rn1*p1)/(rn1*p1 + rn0*(1-p1))
        stdp0 = (0.5-_N.abs(mu_p0-0.5))*0.05 + 0.0001
        istdp0= 1./stdp0
        nC0    = _ss.norm.cdf((1-mu_p0)*istdp0) - _ss.norm.cdf(-mu_p0*istdp0)
        lnC0   = _N.log(nC0)

        lprop1 = -0.5*(ln2pi+_N.log(stdp1*stdp1))-0.5*(p1 - mu_p1)*(p1-mu_p1)*(istdp1*istdp1) - lnC1 + lpmf1 #  forward
        lprop0 = -0.5*(ln2pi+_N.log(stdp0*stdp0))-0.5*(p0 - mu_p0)*(p0-mu_p0)*(istdp0*istdp0) - lnC0 + lpmf0 #  forward

        ll1 = Llklhds(dist, cts, rn1, p1)  #  forward

        lrat = ll1 - ll0 + lprop0 - lprop1

        lrats[it] = lrat
        llklhd0[it]  = ll0
        llklhd1[it]  = ll1
        lpprs1[it]    = lprop1
        lpprs0[it]    = lprop0

        aln   = 1 if (lrat > 0) else _N.exp(lrat)
        #aln   = 1 if (lrat > 0) else 0#else _N.exp(lrat)
        accpts[it] = ran_accpt[it] < aln
        if accpts[it]:   #  accept
            u0 = u1
            rn0 = rn1
            ll0 = ll1
            p0  = p1

        dty[it] = dist

    return prop_ps, prop_rns
