import sampling_helper_V2 as sh
from matplotlib import pyplot as plt 
import numpy as np

MG = sh.MapGenerator('../map_.npy')
MG.loadMap()
MG.getShoreLine()
contour, contour_map = MG.drawLineFromShore(dist=2.5, color=(235,52,210))
x,y,z,x1,y1 = MG.interpolateLine(contour[1])

PS = sh.ParkingSampler()

fig, axs = plt.subplots(4, 4)
fig.suptitle('Spawn generation')
fig.set_figheight(11)
fig.set_figwidth(11)
for i in range(16):
    bx,by,brz = PS.drawBuoyPosition(x,y,z)
    axs[i%4,int(i/4)].plot(x,y)
    axs[i%4,int(i/4)].plot(MG.fine_contour[0][:,0,0], MG.fine_contour[0][:,0,1], 'r')
    axs[i%4,int(i/4)].set_xlim([bx-150,bx+150])
    axs[i%4,int(i/4)].set_ylim([by-150,by+150])
    axs[i%4,int(i/4)].quiver(bx,by,np.cos(brz),np.sin(brz), color='m', scale=10)
    for j in range(100):                                               
        ux, uy, urz = PS.drawUSVPosition(bx,by,brz)
        axs[i%4,int(i/4)].quiver(ux,uy,np.cos(urz),np.sin(urz))
fig.tight_layout(rect=[0,0.03,1,0.95])
plt.show()


