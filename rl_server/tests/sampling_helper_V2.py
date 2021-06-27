import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

def euler2quat(yaw):
    ''' CONVERTS THE RZ (or YAW) INTO A WELL FORMED QUATERNION
    rz: the yaw in radiant
    '''
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    qx = 0
    qy = 0
    qz = sy
    qw = cy
    return np.array([qx,qy,qz,qw])

def degrees2rad(degrees):
    rad = np.pi*degrees/180
    return rad

def rad2degrees(rad):
    degrees = 180*rad/np.pi
    return degrees

class MapGenerator:
    def __init__(self, path, res=10, offset = [1000, 3000]):
        self.path = path
        self.res = res # pixels per meters
        self.offset = offset
    
    def loadMap(self):
        # LOAD MAP
        map_ = (np.load(self.path)*255).astype(np.uint8)
        map_visu = np.zeros((map_.shape[0],map_.shape[1],3),dtype=np.uint8)
    
        # MOVE TO CORRECT PROJECTION
        map_visu = cv2.rotate(map_visu, cv2.ROTATE_90_COUNTERCLOCKWISE)
        map_ = cv2.rotate(map_, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.map_visu = cv2.flip(map_visu, 0)
        self.map_ = cv2.flip(map_, 0)

    def drawLineFromShore(self, dist=9, color=(255,204,0), add_to_visu=True, thick=1):
        # Distance in meters
        distance_contour_map = np.zeros_like(self.map_,dtype=np.uint8)
        ditance_contour_map = cv2.drawContours(distance_contour_map, self.fine_contour, 0, (255), int(dist*2*self.res))
        distance_contour, fine_h = cv2.findContours(distance_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if add_to_visu:
            self.map_visu = cv2.drawContours(self.map_visu, distance_contour, 1, color, thick)
        return distance_contour, distance_contour_map
        
    def getShoreLine(self):
        # CREATE KERNEL, AND INFLATE IMAGE (REMOVE HOLES)
        kernel = np.ones((3,3),np.uint8)
        map_dil_k3_t5 = cv2.dilate(self.map_, kernel, iterations=5)
        inv_dil_map = ((map_dil_k3_t5 == 0)*255).astype(np.uint8)
        
        # REMOVE THE CENTER ISLAND
        blobs = cv2.connectedComponents(inv_dil_map)
        inside = ((blobs[1]==2)*255).astype(np.uint8)
        inside_inv = ((inside == 0)*255).astype(np.uint8)
        
        # GET THE INSIDE OF THE LAKE
        blobs_inside = cv2.connectedComponents(inside_inv)
        clean_inside = (((blobs_inside[1]==1) == 0)*255).astype(np.uint8)

        # COMPUTE CONTOUR (ON INFLATED MAP)
        contours, hierarchy = cv2.findContours(clean_inside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.map_visu = cv2.drawContours(self.map_visu, contours, 0, (0,255,0), 8)
        self.map_visu[:,:,0] = self.map_
        
        # COMPUTE CONTOUR WITH COMPENSATION FOR INFLATION OFFSET
        fine_contour_map = np.zeros_like(self.map_,dtype=np.uint8)
        fine_contour_map = cv2.drawContours(fine_contour_map, contours, 0, (255), 8)
        self.fine_contour, fine_h = cv2.findContours(fine_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.map_visu = cv2.drawContours(self.map_visu, self.fine_contour,0, (0,0,255),1)

    def interpolateLine(self, contour):
        x = contour[:,0, 0]
        y = contour[:,0, 1]
        tck, u = interpolate.splprep([x, y], s=0)
        unew = np.arange(0.002, 0.998, 0.001) 
        out = interpolate.splev(unew, tck)
        t = np.arange(unew.shape[0])
        fx = UnivariateSpline(t, out[0], k=4)
        fy = UnivariateSpline(t, out[1], k=4)
        t2 = np.arange(unew.shape[0]*4)/4
        x = fx(t2)
        y = fy(t2)
        x1 = fx.derivative(1)(t2)
        y1 = fy.derivative(1)(t2)
        rz = np.arctan2(y1, x1) + np.pi/2
        return (x - self.offset[0])/self.res, (y-self.offset[1])/self.res, rz, x1, y1

class ParkingSampler:
    def __init__(self, buoy_dist=2.5, usv_spawn_cone=110, usv_view_angle=110, min_buoy_usv=5.0, max_buoy_usv=10.0, res=1):
        self.buoy_dist = buoy_dist
        self.usv_spawn_cone = usv_spawn_cone
        self.usv_view_angle = usv_view_angle
        self.min_buoy_usv = min_buoy_usv*res
        self.max_buoy_usv = max_buoy_usv*res
        self.buoy_spawn_noise = 15.

    def drawBuoyPosition(self, x, y, rz):
        L= []
        for idx in range(rz.shape[0]):
            if idx > 20:
                L.append(np.std(rz[idx-20:idx+20]))
            else: 
                L.append(np.std(rz[:idx+20]))
        c = (np.array(L) < 0.4)*1.0
        idx = np.random.choice(np.arange(x.shape[0]), p = c/np.sum(c))
        buoy_heading = rad2degrees(rz[idx]) + self.buoy_spawn_noise/2. - self.buoy_spawn_noise*np.random.rand()
        buoy_heading = degrees2rad(buoy_heading)
        return x[idx], y[idx],  buoy_heading

    def spawnBuoyAndUSV(self):
        buoy_pose_x, buoy_pose_y, buoy_heading = self.drawBuoyPosition(x,y,rz)
        usv_pose_x, usv_pose_y, usv_heading = self.drawUSVPosition(buoy_pose_x, buoy_pose_y, buoy_heading)
        buoy = [buoy_pose_x, buoy_pose_y, buoy_heading]
        usv = [usv_pose_x, usv_pose_y, usv_heading]
        return buoy, usv

    def drawUSVPosition(self, buoy_pose_x, buoy_pose_y, buoy_heading):
        # draw from polar coordinates
        r = np.random.rand()*(self.max_buoy_usv - self.min_buoy_usv) + self.min_buoy_usv
        theta = rad2degrees(buoy_heading) + self.usv_spawn_cone/2. - np.random.rand()*self.usv_spawn_cone 
        theta = degrees2rad(theta)
        # convert to cartesian and translate to world frame
        usv_pose_x = np.cos(theta)*r + buoy_pose_x
        usv_pose_y = np.sin(theta)*r + buoy_pose_y
        # draw usv heading
        opt_ang_wf = np.arctan2(usv_pose_y - buoy_pose_y, usv_pose_x - buoy_pose_x)
        usv_heading = opt_ang_wf + degrees2rad(self.usv_view_angle/2. - self.usv_view_angle*np.random.rand()) + np.pi
        return usv_pose_x, usv_pose_y, usv_heading
