#  j(u)
import mcmcFigs as mF

N = 800
xL = -10
xH = -2
u = _N.linspace(xL, xH, N)

u1= -6.5
u2= -4
um= (u1+u2)*0.5    #  jx(um) = 50%

mag= 4./(u2-u1)

ut = (u - um)*mag   #  transformed u

j  = 1 / (1 + _N.exp(2*ut))

ilo = _N.where((j[1:] < 0.9999) & (j[:-1] >= 0.9999))[0]
ihi = _N.where((j[1:] < 1e-4) & (j[:-1] >= 1e-4))[0]

uP1m4  = _N.log(0.0001/(1-0.0001))    #  corresponding val of p=1e-3
uP1m3  = _N.log(0.001/(1-0.001))    #  corresponding val of p=1e-3
uP1m2  = _N.log(0.01/(1-0.01))    #  corresponding val of p=1e-2
uP5m2  = _N.log(0.05/(1-0.05))    #  corresponding val of p=5e-2

iP1m4 = _N.where((u[1:] > uP1m4) & (u[:-1] <= uP1m4))[0]
iP1m3 = _N.where((u[1:] > uP1m3) & (u[:-1] <= uP1m3))[0]
iP1m2 = _N.where((u[1:] > uP1m2) & (u[:-1] <= uP1m2))[0]
iP5m2 = _N.where((u[1:] > uP5m2) & (u[:-1] <= uP5m2))[0]

yoff  = 1.7

fig   = _plt.figure(figsize=(8, 7))
ax    = fig.add_subplot(1, 1, 1)
_plt.plot(u, j, color="black")
_plt.plot(u, j+yoff, color="black")
_plt.ylim(-0.1, 2.8)
_plt.xlim(xL, xH)
_plt.plot([u[ilo], u[ilo]], [j[ilo], j[ilo]], marker="x", ms=14, color="blue")
_plt.plot([u[ihi], u[ihi]], [j[ihi], j[ihi]], marker="x", ms=14, color="blue")

_plt.plot([u[ilo], u[ilo]], [j[ilo]+yoff, j[ilo]+yoff], marker="x", ms=14, color="blue")
_plt.plot([u[ihi], u[ihi]], [j[ihi]+yoff, j[ihi]+yoff], marker="x", ms=14, color="blue")


_plt.plot([u[iP1m4], u[iP1m4]], [j[iP1m4], j[iP1m4]], marker=".", ms=12, color="black")
_plt.text(u[iP1m4]+0.2, j[iP1m4]+0.02, "0.0001")
_plt.plot([u[iP1m4], u[iP1m4]], [j[iP1m4]+yoff, j[iP1m4]+yoff], marker=".", ms=12, color="black")
_plt.text(u[iP1m4]+0.2, j[iP1m4]+yoff+0.02, "0.0001")


_plt.plot([u[iP1m3], u[iP1m3]], [j[iP1m3], j[iP1m3]], marker=".", ms=12, color="black")
_plt.text(u[iP1m3]+0.2, j[iP1m3]+0.02, "0.001")
_plt.plot([u[iP1m3], u[iP1m3]], [j[iP1m3]+yoff, j[iP1m3]+yoff], marker=".", ms=12, color="black")
_plt.text(u[iP1m3]+0.2, j[iP1m3]+yoff+0.02, "0.001")



_plt.plot([u[iP1m2], u[iP1m2]], [j[iP1m2], j[iP1m2]], marker=".", ms=12, color="black")
_plt.text(u[iP1m2]+0.2, j[iP1m2]+0.02, "0.01")
_plt.plot([u[iP1m2], u[iP1m2]], [j[iP1m2]+yoff, j[iP1m2]+yoff], marker=".", ms=12, color="black")
_plt.text(u[iP1m2]+0.2, j[iP1m2]+yoff+0.02, "0.01")

_plt.plot([u[iP5m2], u[iP5m2]], [j[iP5m2], j[iP5m2]], marker=".", ms=12, color="black")
_plt.text(u[iP5m2]+0.2, j[iP5m2]+0.02, "0.05")
_plt.plot([u[iP5m2], u[iP5m2]], [j[iP5m2]+yoff, j[iP5m2]+yoff], marker=".", ms=12, color="black")
_plt.text(u[iP5m2]+0.2, j[iP5m2]+yoff+0.02, "0.05")


mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(xlabel="u", ylabel="j(u)", yticks=[0, 0.5, 1, yoff, yoff+0.5, yoff+1], yticksD=["0", "0.5", "1", "0", "0.5", "1"], xlim=None, ylim=None, tickFS=18, labelFS=20)
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95)
_plt.savefig("jx.pdf")
