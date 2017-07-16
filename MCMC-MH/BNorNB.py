import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt
import time as _tm
import scipy.stats as _ss
import os

import warnings
warnings.filterwarnings('error')

#  connection between 
logfact= None

uTH1= -6.5
uTH2= -3

uTHa= -6.5
uTHb= -3

ints = _N.arange(20000)
ln2pi= 1.8378770664093453

mn_u = 0
iu_sd2 = (1/1.2)*(1/1.2)
#mn_u = -1
#iu_sd2 = (1/1.05)*(1/1.05)

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


def BNorNBonly(GibbsIter, iters, w, j, u0, rn0, dist, cts, rns, us, dty, xn, jxs, mvtyps, lls, accpts, llklhd0, llklhd1, lrats, lpprs0, lpprs1, stdu=0.03, propratm=0):
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
    rnmin = _N.array([-1, _N.max(cts)+1, 1], dtype=_N.int)

    rdis= _N.random.rand(iters)  #
    rdns = _N.random.randn(iters)
    ran_accpt  = _N.random.rand(iters)

    p0  = 1./(1 + _N.exp(-u0))

    ll0 = Llklhds(dist, cts, rn0, p0)
    #  the poisson distribution needs to be truncated
    prop_us = _N.empty(iters)
    prop_rns = _N.empty(iters, dtype=_N.int)
    prop_dty = _N.empty(iters, dtype=_N.int)

    zr2rnmins  = _N.array([None, _N.arange(rnmin[1]), _N.arange(rnmin[2])]) # rnmin is a valid value for n

    u_m     = (uTH1+uTH2)*0.5
    mag     = 4./(uTH2 - uTH1)

    r_prior = 1
    p_prior = 0.9996

    ks = _N.arange(500000)
    lpriorpmf = logfact[ks + r_prior - 1] - logfact[r_prior-1] - logfact[ks] + r_prior*_N.log(1-p_prior) + ks*_N.log(p_prior)   #  prior

    for it in xrange(iters):
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted
        lls[it] = ll0
        dty[it] = dist

        ut    = (u0 - u_m)*mag
        jx    = 0.9 / (1 + _N.exp(2*ut))
        #
        #dbtt1 = _tm.time()
        jxs[it] = jx

        if it % 1000 == 0:
            print it

        if _N.random.rand() < jx:  #  JUMP
            todist = _cd.__NBML__ if dist == _cd.__BNML__ else _cd.__BNML__
            mv = 0
            mvtyps[it] = 1
            #  jump
            #print "here  %(it)d   %(jx).3f" % {"it" : it, "jx" : jx}
            u1 = (uTHa + uTHb) - u0
            prop_us[it] = u1
            prop_dty[it]= todist
            ip1 = 1 + _N.exp(-u1)
            p0  = 1 / (1 + _N.exp(-u0))
            p1  = 1 / (1 + _N.exp(-u1))
            p1x = 1 / (1 + _N.exp(-(u1+xn)))

            #rr   = rn0*p0/p1 
            rr   = ((rn0*p0) / (p1*(1-p0))) if dist == _cd.__NBML__ else ((rn0*p0*(1-p1))/p1)
            irr  = int(rr)
            rmdr = rr-irr
            rn1   = int(rr)

            if _N.random.rand() < rmdr:
                rn1 += 1
            prop_rns[it] = rn1

            #print "B4  rn0 %(0)d   rn1 %(1)d  (%(1f).8e   %(2f).8e)   p0 %(p0).4e  p1 %(p1).4e" % {"0" : rn0, "1" : rn1, "p0" : p0, "p1" : p1, "1f" : (rn0 * p0)/p1, "2f" : (p0/p1)}

            # if _N.random.rand() < rmdr:
            #     rn1  += 1
            #print "AFT  rn0 %(0)d   rn1 %(1)d  (%(1f).8e   %(2f).8e)   p0 %(p0).4e  p1 %(p1).4e" % {"0" : rn0, "1" : rn1, "p0" : p0, "p1" : p1, "1f" : (rn0 * p0)/p1, "2f" : (p0/p1)}

            #print "%(it)d   u0  %(1).3e  %(2).3e" % {"1" : u0, "2" : u1, "it" : it}
            #print "%d  propose jump" % it

            p1x = 1 / (1 + _N.exp(-(u1+xn)))
            #lpPR   = _N.log((uTH1 - u0) / (uTH1 - u1))         #  deterministic crossing.  Jac = 1
            utr = (u1 - u_m)*mag
            #utr = (uTH1 + uTH2)-u1
            jxr = 1 / (1 + _N.exp(2*utr))
            #lpPR   = _N.log(jxr/jx)
            if dist == _cd.__BNML__:
                ljac = _N.log((jxr/jx) * (p0 / (p1*(1-p1))))
            else:
                ljac = _N.log((jxr/jx) * ((p0*(1-p0)) / p1))
            lprop0 = lprop1 = 0  #

            lpru0   = 0
            lpru1   = 0
            lpru0   = 0
            lpru1   = 0


            #ljac   = _N.log((jxr/jx) * (p0/p1))
        else:    #  ########   DIFFUSION    ############
            mvtyps[it] = 0
            todist = dist
            prop_dty[it]= todist

            zr2rnmin = zr2rnmins[dist]
            ljac    = 0

            ##  propose an rn1 
            m2          = 1./rn0 + 0.9      # rn0 is the mean for proposal for rn1
            p_prp_rn1        = 1 - 1./(rn0*m2)  # param p for proposal for rn1
            r_prp_rn1        = rn0 / (rn0*m2-1) # param r for proposal for rn1
            ir_prp_rn1 = int(r_prp_rn1)

            bGood = False   #  rejection sampling of rn1
            while not bGood:
                rn1 = _N.random.negative_binomial(ir_prp_rn1, 1-p_prp_rn1)
                if rn1 >= rnmin[dist]:
                    bGood = True
            prop_rns[it] = rn1

            #########  log proposal density for rn1
            ir_prp_rn1 = int(r_prp_rn1)
            ltrms = logfact[zr2rnmin+ir_prp_rn1-1]  - logfact[ir_prp_rn1-1] - logfact[zr2rnmin] + ir_prp_rn1*_N.log(1-p_prp_rn1) + zr2rnmin*_N.log(p_prp_rn1)
            lCnb1        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated pmf

            lpmf1       = logfact[rn1+ir_prp_rn1-1]  - logfact[ir_prp_rn1-1] - logfact[rn1] + r_prp_rn1*_N.log(1-p_prp_rn1) + rn1*_N.log(p_prp_rn1) - lCnb1

            #########  log proposal density for rn0
            ##  rn1
            m2          = 1./rn1 + 0.9      # rn0 is the mean for proposal for rn1
            p_prp_rn0        = 1 - 1./(rn1*m2)  # param p for proposal for rn1
            r_prp_rn0        = rn1 / (rn1*m2-1.) # param r for proposal for rn1
            ir_prp_rn0 = int(r_prp_rn0)

            ltrms = logfact[zr2rnmin+ir_prp_rn0-1]  - logfact[ir_prp_rn0-1] - logfact[zr2rnmin] + ir_prp_rn0*_N.log(1-p_prp_rn0) + zr2rnmin*_N.log(p_prp_rn0)
            smelt = _N.sum(_N.exp(ltrms))
            # print "------"
            # print "%.5f" % smelt
            # print dist
            # print todist
            # print ir_prp_rn0
            # print p_prp_rn0
            # print zr2rnmin
            lCnb0        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated 

            lpmf0       = logfact[rn0+ir_prp_rn0-1]  - logfact[ir_prp_rn0-1] - logfact[rn0] + r_prp_rn0*_N.log(1-p_prp_rn0) + rn0*_N.log(p_prp_rn0) - lCnb0

            ###################################################
            #  propose p1
            #  sample using p from [0, 0.75]  mean 0.25, sqrt(variance) = 0.25
            #  p1 x rn1 = p0 x rn0      --> p1 = p0 x rn0/rn1   BINOMIAL
            #  

            try:
                mn_u1 = -_N.log(rn1 * (1+_N.exp(-u0))/rn0 - 1) if dist == _cd.__BNML__ else (u0 - _N.log(float(rn1)/rn0))
            except Warning:
                print "todist   %(to)d   dist %(fr)d" % {"to" : todist, "fr" : dist}
                print "%.5e" % (rn1 * (1+_N.exp(-u0))/rn0 - 1)
                print "%.5e" % (u0 - _N.log(float(rn1)/rn0))


            u1          = mn_u1 + stdu*rdns[it]
            p1x = 1 / (1 + _N.exp(-(u1+xn)))
            p1  = 1/(1 + _N.exp(-u1))

            prop_us[it] = u1
            mn_u0 = -_N.log(rn0 * (1+_N.exp(-u1))/rn1 - 1) if dist == _cd.__BNML__ else (u1 - _N.log(float(rn0)/rn1))

            lprop1 = -0.5*(u1 - mn_u1)*(u1-mn_u1)*istdu2 + lpmf1 #  forward
            lprop0 = -0.5*(u0 - mn_u0)*(u0-mn_u0)*istdu2 + lpmf0 #  backwards

            lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2
            lpru1   = -0.5*(u1 - mn_u)*(u1 - mn_u)*iu_sd2
        ll1 = Llklhds(todist, cts, rn1, p1)  #  forward

        lrat = ll1 - ll0 + lpru1 - lpru0 + lpriorpmf[rn1 + rnmin[todist]] - lpriorpmf[rn0 + rnmin[dist]] + lprop0 - lprop1 + ljac
        #lrat = ll1 - ll0 + lpriorpmf[rn1 + rnmin[todist]] - lpriorpmf[rn0 + rnmin[dist]] + lprop0 - lprop1 + ljac

        lrats[it] = lrat
        #ljacs[it] = ljac
        llklhd0[it]  = ll0
        llklhd1[it]  = ll1
        lpprs1[it]    = lprop1
        lpprs0[it]    = lprop0

        aln   = 1 if (lrat > 0) else _N.exp(lrat)
        #aln   = 1 if (lrat > 0) else 0else _N.exp(lrat)
        accpts[it] = ran_accpt[it] < aln
        if accpts[it]:   #  accept
            u0 = u1
            rn0 = rn1
            ll0 = ll1
            p0  = p1
            dist=todist

    return prop_us, prop_rns, prop_dty
