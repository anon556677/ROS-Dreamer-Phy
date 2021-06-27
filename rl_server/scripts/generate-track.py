import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, splprep, splev

class MakeTrack:
    def __init__(self, path, generate_sdf=False, from_object=False):
        if from_object:
            self.loadSave()
        else:
            self.path = path
            self.loadTrack()
        
    def loadTrack(self):
        return cv2.imread(self.path,-1)

    def processTrackImage(self, raw_track):
        track = raw_track[:,:,3] > 100
        grey_mask = (raw_track[:,:,0] > 100)*(raw_track[:,:,0] < 160)
        track[grey_mask] = 0
        kernel = np.ones((5, 5), np.uint8)
        track = cv2.dilate(track.astype(np.uint8), kernel, cv2.BORDER_REFLECT)
        clean_track = cv2.erode(track, kernel, cv2.BORDER_REFLECT)
        return clean_track

    def getContours(self, clean_track):
        contours, _ = cv2.findContours(clean_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inside = contours[1]
        outside = contours[0]
        return inside[:,0,:], outside[:,0,:]

    def rescaleContours(self, contour):
        contour = contour.astype(np.float32)
        contour[:,0] = contour[:,0]*self.factor + self.offset[0]*self.factor
        contour[:,1] = -contour[:,1]*self.factor - self.offset[1]*self.factor
        return contour

    def fitSpline(self, contour):
        x = contour[:, 0]
        y = contour[:, 1]
        tck, u = splprep([x, y],s = self.smoothing_factor, per=True)
        unew = np.arange(0.000, 1.0, self.cone_ditance)
        out = splev(unew, tck)

        return out[0], out[1]

    def interpolateLine(x, y, theta='normal'):
        t = np.arange(x.shape[0])
        fx = UnivariateSpline(t, x, k=4, s=0)
        fy = UnivariateSpline(t, y, k=4, s=0)
        t2 = np.arange(x.shape[0]*4)/4
        x = fx(t2)
        y = fy(t2)
        x1 = fx.derivative(1)(t2)
        y1 = fy.derivative(1)(t2)

        if theta == 'normal':
            rz = np.arctan2(y1, x1) + np.pi/2
        elif theta == 'tangent':
            rz = np.arctan2(y1, x1)

        x = x + np.cos(rz)*0.5
        y = y + np.sin(rz)*0.5
        return x, y, rz, x1, y1

def makeCone(px, py, id):
    cone = ['<include>',
    '    <uri>model://racecar_description/models/cone</uri>',
    '    <pose>'+str(px)+' '+str(py)+' 0 0 0 0</pose>',
    '    <name>cone_'+str(id)+'</name>',
    '</include>']
    return cone

def genSDF(contours, name='nurburgring'):
    sdf_header = ['<?xml version="1.0" ?>',
                '<sdf version="1.5">',
                '<world name="'+name+'">']
    ground = ['<model name="ground_plane">',
        '    <static>true</static>',
        '    <link name="link">',
        '    <collision name="collision">',
        '        <geometry>',
        '        <plane>',
        '            <normal>0 0 1</normal>',
        '            <size>150 150</size>',
        '        </plane>',
        '        </geometry>',
        '        <surface>',
        '        <friction>',
        '            <ode>',
        '            <mu>100</mu>',
        '            <mu2>50</mu2>',
        '            </ode>',
        '        </friction>',
        '        </surface>',
        '    </collision>',
        '   <visual name="visual">',
        '        <cast_shadows>false</cast_shadows>',
        '        <geometry>',
        '        <plane>',
        '            <normal>0 0 1</normal>',
        '            <size>150 150</size>',
        '        </plane>',
        '        </geometry>',
        '        <material>',
        '        <script>',
        '            <uri>file://media/materials/scripts/gazebo.material</uri>',
        '            <name>Gazebo/Grey</name>',
        '        </script>',
        '        </material>',
        '    </visual>',
        '    </link>',
        '</model>']
    sun = ['<scene>',
        '    <ambient>0.75 0.75 0.75 1.0</ambient>',
        '</scene>',
        ' ',
        '<include>',
        '    <uri>model://sun</uri>',
        '</include>']
    sdf_footer = ['</world>',
        '</sdf>']
    cone_id = 0
    world = sdf_header + sun + ground
    for contour in contours:
        for point in contour:
            world += makeCone(point[0], point[1], cone_id)
            cone_id += 1
    world += sdf_footer
    return world

def dump_world(world, name='nurburgring'):
    with open(name+'.world', 'w') as f:
        for line in world:
            f.write("%s\n" % line)


factor = 0.07
raw_track = loadTrack('nurburgring.png')
#plt.figure()
#plt.imshow(raw_track)
clean_track_image = processTrackImage(raw_track)
#plt.figure()
#plt.imshow(clean_track_image)
#plt.figure()
inside, outside = getContours(clean_track_image)
#plt.plot(outside[:,0],outside[:,1],label='outside')
#plt.plot(inside[:,0],inside[:,1],label='inside')
offset_x = (np.max(outside[:,0]) - np.min(outside[:,0]))/2.0 + np.min(outside[:,0])
offset_y = (np.max(outside[:,1]) - np.min(outside[:,1]))/2.0 + np.min(outside[:,1])
offset = [-offset_x, -offset_y]
print(offset)
inside = rescaleContours(inside, factor, offset = offset)
outside = rescaleContours(outside, factor, offset = offset)
#plt.figure()
#plt.plot(outside[:,0],outside[:,1],label='outside')
#plt.plot(inside[:,0],inside[:,1],label='inside')
inside_x, inside_y = fitSpline(inside)
outside_x , outside_y = fitSpline(outside)
x2,y2,rz2,x21,y21 = interpolateLine(outside_x,outside_y)
