import cv2
import numpy as np
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

def rsign():
    return np.sign(np.random.rand() - 0.5)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

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
    
    def loadMap(self, with_visu=False):
        # LOAD MAP
        map_ = (np.load(self.path)*255).astype(np.uint8)
        if with_visu:
            map_visu = np.zeros((map_.shape[0],map_.shape[1],3),dtype=np.uint8)
            # MOVE TO CORRECT PROJECTION
            map_visu = cv2.rotate(map_visu, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.map_visu = cv2.flip(map_visu, 0)
        # MOVE TO CORRECT PROJECTION
        map_ = cv2.rotate(map_, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.map_ = cv2.flip(map_, 0)

    def drawLineFromShore(self, dist=9, color=(255,204,0), add_to_visu=False, thick=1):
        # Distance in meters
        distance_contour_map = np.zeros_like(self.map_,dtype=np.uint8)
        ditance_contour_map = cv2.drawContours(distance_contour_map, self.fine_contour, 0, (255), int(dist*2*self.res))
        distance_contour, fine_h = cv2.findContours(distance_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if add_to_visu:
            self.map_visu = cv2.drawContours(self.map_visu, distance_contour, 1, color, thick)
        return distance_contour, distance_contour_map
        
    def getShoreLine(self, with_visu=False):
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
        if with_visu:
            self.map_visu = cv2.drawContours(self.map_visu, contours, 0, (0,255,0), 8)
            self.map_visu[:,:,0] = self.map_
        
        # COMPUTE CONTOUR WITH COMPENSATION FOR INFLATION OFFSET
        fine_contour_map = np.zeros_like(self.map_,dtype=np.uint8)
        fine_contour_map = cv2.drawContours(fine_contour_map, contours, 0, (255), 8)
        self.fine_contour, fine_h = cv2.findContours(fine_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if with_visu:
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
        rz = np.arctan2(y1, x1)
        return (x - self.offset[0])/self.res, (y-self.offset[1])/self.res, rz, x1, y1

class FollowingSampler:
    def __init__(self, path_to_data, ideal_dist, min_dist, max_dist, warmup, target_step, alpha):
        self.MG = MapGenerator(path_to_data)
        self.MG.loadMap()
        self.MG.getShoreLine()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.ideal_dist = ideal_dist
        self.warmup = warmup
        self.target_step = target_step
        self.alpha = alpha
        self.running_reward = None
        self.getIdealPoses()

    def sample(self, step, reward, training, mode):
        if mode == 'none':
            x, y, rz = self.sampleRandomIdealPosition()
            t = np.zeros(10)
            p = np.ones(t.shape[0])
            return [x,y,rz], t, p/np.sum(p)
        elif mode == 'random':
            x, y, rz, t, p = self.samplePositionFixedCurriculum(step=self.target_step+1)
            return [x,y,rz], t, p
        else:
            x, y, rz, t, p = self.samplePositionFixedCurriculum(step=step, mode=mode)
            return [x,y,rz], t, p

    def getIdealPoses(self):
        ideal, _ = self.MG.drawLineFromShore(dist = self.ideal_dist, thick = 1)
        #print(len(ideal))

        length = []
        idxs = []
        for idx, i in enumerate(ideal):
            length.append(cv2.arcLength(i, True))
            idxs.append(idx)

        sort = sorted(range(len(length)), key=lambda k: length[k])
        #print(sort[-2])

        self.ideal_x, self.ideal_y, self.ideal_rz, x1, y1 = self.MG.interpolateLine(ideal[sort[-2]])

    def sampleRandomIdealPosition(self):
        idx = np.random.choice(np.arange(self.ideal_x.shape[0]))
        return self.ideal_x[idx], self.ideal_y[idx], self.ideal_rz[idx]
    
    def samplePositionFixedCurriculum(self, step, mode='power', max_ang_noise=np.pi/2):
        px, py, rz = self.sampleRandomIdealPosition()
        if step > self.target_step:
            coeff = 1.0
        elif step > self.warmup:
            if mode=='sigmoid':
                t = (step-self.warmup)/(self.target_step-self.warmup)*6 - 3
                coeff = np.tanh(t)/2 + 0.5
            elif mode=='power':
                max_v = 5**self.alpha
                t = (step-self.warmup)/(self.target_step-self.warmup)*5
                coeff = t**self.alpha/max_v
            else:
                raise ValueError('Unknown mode: '+mode)
        else:
            t = np.zeros(10)
            p = np.ones(t.shape[0])
            return px, py, rz, t, p/np.sum(p) 
        # get distance distribution
        dist, dist_dist = self.genDualGaussianDist(coeff)
        # apply distance to usv
        dist = np.random.choice(dist,1,p=dist_dist)
        px = px + np.cos(rz+np.pi/2)*dist
        py = py + np.sin(rz+np.pi/2)*dist
        # compensate for the angular error created
        rz = rz + np.arctan2(-dist,5)
        # add some noise on the ideal yaw
        t, p = self.genGaussian(coeff)
        yaw_c = np.random.choice(t,1,p=p)
        rz = rz + yaw_c*max_ang_noise*rsign()
        return px[0], py[0], rz[0], t, p

    def genDualGaussianDist(self, coeff, sig1 = 1., sig2 = 0.5, eps = 0.025):
        bmax = self.max_dist - self.ideal_dist
        bmin = self.min_dist - self.ideal_dist
        t = np.arange(bmin, bmax, 0.01)
        mu1 = coeff * bmin
        mu2 = coeff * bmax
        pbmin = int((mu1 - bmin)/0.01)
        pbmax = int((mu2 - bmin)/0.01)
        g1 = gaussian(t, mu1, sig1)
        g2 = gaussian(t, mu2, sig2)
        w = g1 + g2
        w[pbmin:pbmax] = (w[pbmin:pbmax] < eps)*eps + (w[pbmin:pbmax] > eps)*w[pbmin:pbmax]
        p = w/np.sum(w)
        return t, p

    def genGaussian(self, coeff, sig = 0.25):
        t = np.arange(0,1,0.01)
        g = gaussian(t, coeff, sig)
        return t, g/np.sum(g)


class ParkingSampler:
    def __init__(self, path_to_data, buoy_dist=2.5, usv_spawn_cone=110, usv_view_angle=110, min_buoy_usv=5.0, max_buoy_usv=8.0, res=1, target_step=1e6, warmup=0.25e6, alpha=1.0):
        self.buoy_dist = buoy_dist
        self.usv_spawn_cone = usv_spawn_cone
        self.usv_view_angle = usv_view_angle
        self.min_buoy_usv = min_buoy_usv*res
        self.max_buoy_usv = max_buoy_usv*res
        self.buoy_spawn_noise = 15.
        self.target_step = target_step
        self.warmup = warmup
        self.alpha = alpha
        self.MG = MapGenerator(path_to_data)
        self.MG.loadMap()
        self.MG.getShoreLine()
        contour, _ = self.MG.drawLineFromShore(dist=2.5, color=(235,52,210))
        self.x, self.y, self.rz, _, _ = self.MG.interpolateLine(contour[1])

    def drawBuoyPosition(self, x, y, rz):
        rz = rz + np.pi/2
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

    def sampleBuoyAndUSV(self, step, mode, pose):
        buoy = self.drawBuoyPosition(self.x, self.y, self.rz)
        if mode == 'none':
            usv = self.drawUSVPosition(pose)
            t = np.zeros(10)
            p = np.ones(t.shape[0])
            p = p/np.sum(p)
        else:
            usv, t, p = self.drawUSVPositionCurriculum(step, mode, buoy)
        return buoy, usv, t ,p
    
    def genGaussian(self, coeff, sig = 0.25):
        t = np.arange(0, 1, 0.01)
        g = gaussian(t, coeff, sig)
        return t, g/np.sum(g)
    
    def drawUSVPosition(self, pose):
        # draw from polar coordinates
        r = np.random.rand()*(self.max_buoy_usv -self.min_buoy_usv)
        theta = rad2degrees(pose[2]) + self.usv_spawn_cone/2. - np.random.rand()*self.usv_spawn_cone
        theta = degrees2rad(theta)
        # convert to cartesian and translate to world frame
        usv_pose_x = np.cos(theta)*r + pose[0]
        usv_pose_y = np.sin(theta)*r + pose[1]
        # draw usv heading
        opt_ang_wf = np.arctan2(usv_pose_y - pose[1], usv_pose_x - pose[0])
        ang_noise = (np.random.rand()*self.usv_view_angle/2)*rsign()
        usv_heading = opt_ang_wf + degrees2rad(ang_noise) + np.pi
        return usv_pose_x, usv_pose_y, usv_heading


    def drawUSVPositionCurriculum(self, step, mode, pose):
        # Compute advancement in the training
        if step > self.target_step:
            coeff = 1.0
        elif step > self.warmup:
            if mode=='sigmoid':
                t = (step-self.warmup)/(self.target_step-self.warmup)*6 - 3
                coeff = np.tanh(t)/2 + 0.5
            elif mode=='power':
                max_v = 5**self.alpha
                t = (step-self.warmup)/(self.target_step-self.warmup)*5
                coeff = t**self.alpha/max_v
            else:
                raise ValueError('Unknown mode: '+mode)
        else:
            coeff = 0
        
        t, p = self.genGaussian(coeff, sig = coeff/2. + 0.1)
        
        # draw from polar coordinates
        r = np.random.choice(t* self.max_buoy_usv + self.min_buoy_usv, 1, p=p)[0]
        theta = np.random.choice(t * self.usv_spawn_cone/2, 1, p=p)[0]*rsign() + rad2degrees(pose[2])
        theta = degrees2rad(theta)
        # convert to cartesian and translate to world frame
        usv_pose_x = np.cos(theta)*r + pose[0]
        usv_pose_y = np.sin(theta)*r + pose[1]
        # draw usv heading
        opt_ang_wf = np.arctan2(usv_pose_y - pose[1], usv_pose_x - pose[0])
        ang_noise = np.random.choice(t*self.usv_view_angle/2,1,p=p)[0]*rsign()
        usv_heading = opt_ang_wf + degrees2rad(ang_noise) + np.pi
        return [usv_pose_x, usv_pose_y, usv_heading], t, p
