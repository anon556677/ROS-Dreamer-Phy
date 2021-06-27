import sampling_helper_V2 as sh
from matplotlib import pyplot as plt 
import numpy as np

MG = sh.MapGenerator('../map_.npy')
MG.loadMap()
MG.getShoreLine()
contour, contour_map = MG.drawLineFromShore(dist=2.5, color=(235,52,210))
x,y,z,x1,y1 = MG.interpolateLine(contour[1])

PS = sh.ParkingSampler()

dim = 3

fig, axs = plt.subplots(dim, dim)
fig.suptitle('DY reward distribution')
fig.set_figheight(11)
fig.set_figwidth(11)
b = 255
r = np.linspace(0,255,1)

for i in range(dim*dim):
    bx,by,brz = PS.drawBuoyPosition(x,y,z)
    axs[i%dim,int(i/dim)].plot(x,y)
    axs[i%dim,int(i/dim)].plot(MG.fine_contour[0][:,0,0]/10 -100, MG.fine_contour[0][:,0,1]/10 -300, 'r')
    axs[i%dim,int(i/dim)].set_xlim([bx-15,bx+15])
    axs[i%dim,int(i/dim)].set_ylim([by-15,by+15])
    axs[i%dim,int(i/dim)].quiver(bx,by,np.cos(brz),np.sin(brz), color='m', scale=10)
    for j in range(100):                                               
        ux, uy, urz = PS.drawUSVPosition(bx,by,brz)
        dy = np.abs(np.cos(brz)*(uy-by) - np.sin(brz)*(ux-bx))
        axs[i%dim,int(i/dim)].plot(ux,uy, 'o',color=[dy/10,0,1])
fig.tight_layout(rect=[0,0.03,1,0.95])

fig, axs = plt.subplots(dim, dim)
fig.suptitle('DX reward distribution')
fig.set_figheight(11)
fig.set_figwidth(11)
b = 255
r = np.linspace(0,255,1)

for i in range(dim*dim):
    bx,by,brz = PS.drawBuoyPosition(x,y,z)
    axs[i%dim,int(i/dim)].plot(x,y)
    axs[i%dim,int(i/dim)].plot(MG.fine_contour[0][:,0,0]/10 -100, MG.fine_contour[0][:,0,1]/10 -300, 'r')
    axs[i%dim,int(i/dim)].set_xlim([bx-15,bx+15])
    axs[i%dim,int(i/dim)].set_ylim([by-15,by+15])
    axs[i%dim,int(i/dim)].quiver(bx,by,np.cos(brz),np.sin(brz), color='m', scale=10)
    for j in range(100):                                               
        ux, uy, urz = PS.drawUSVPosition(bx,by,brz)
        dy = np.abs(np.sin(brz)*(uy-by) + np.cos(brz)*(ux-bx))
        axs[i%dim,int(i/dim)].plot(ux,uy, 'o',color=[dy/10,0,1])
fig.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
