import mcmcFigs as mF
#  For Reversible Jump MCMC on binomial and negative binomial models,
#  the maximum likelihood is going to be near np or rp/(1-p), but 
#  for small values of p near FF=1, small changes in p can lead to 
#  big changes in n or r in order to keep 1st order statistic, mean, 
#  the same.  Any proposal with mean too far off will be rejected 
#  almost certainly.  For smaller values of p (or more negative u),
#  the amount of variance needed for samples of n and r in order to
#  explore parameter space, gets very large.  We could do something
#  to increase the variance of sampled n or r as p0 gets smaller.  
#  Instead, we go the route of putting a prior on u and r,n to not
#  allow p to get too small.  There is very little difference between
#  models where p=1e-3 and p=1e-7, so we don't really need to explore
#  the parts of the model in which making a proposal distribution is
#  difficult.

#us = -1
us = 0
std= 1.3#.5

tcksz=15
labsz=17

us = us + std*_N.random.randn(20000)
ps = 1/(1 + _N.exp(-us))

fig = _plt.figure(figsize=(5, 10))
ax = fig.add_subplot(3, 1, 1)
_plt.hist(ps, bins=_N.linspace(0, 1, 1001))
mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(xlabel="ps", ylabel="prob", xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=tcksz, labelFS=labsz)

ax = fig.add_subplot(3, 1, 2)
_plt.hist(1-ps, bins=_N.linspace(0, 1, 1001))
mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(xlabel="ps", ylabel="prob", xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=tcksz, labelFS=labsz)

ax = fig.add_subplot(3, 1, 3)
_plt.hist(1/(1-ps), bins=_N.linspace(1, 5, 1001))
mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(xlabel="ps", ylabel="prob", xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=tcksz, labelFS=labsz)

fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.3, hspace=0.3)
