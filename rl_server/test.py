#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def constant_gaussian(x, mu, sig, constant):
    if x < mu-sig:
        return constant
    else:
        return gaussian(x, mu, sig)
def twosigma_gaussian(x, mu, sig_l, sig_r):
    if x < mu:
        return gaussian(x, mu, sig_l)
    else:
        return gaussian(x, mu, sig_r)
# LOAD MAP
map_ = (np.load('map_.npy')*255).astype(np.uint8)
map_visu = np.zeros((map_.shape[0],map_.shape[1],3),dtype=np.uint8)
# MOVE TO CORRECT PROJECTION
map_visu = cv2.rotate(map_visu, cv2.ROTATE_90_COUNTERCLOCKWISE)
map_ = cv2.rotate(map_, cv2.ROTATE_90_COUNTERCLOCKWISE)
map_visu = cv2.flip(map_visu, 0)
map_ = cv2.flip(map_, 0)
# CREATE KERNEL, AND INFLATE IMAGE (REMOVE HOLES)
kernel = np.ones((3,3),np.uint8)
map_dil_k3_t5 = cv2.dilate(map_, kernel, iterations=5)
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
map_visu = cv2.drawContours(map_visu, contours, 0, (0,255,0), 8)
map_visu[:,:,0] = map_
# COMPUTE CONTOUR WITH COMPENSATION FOR INFLATION OFFSET
fine_contour_map = (map_.copy()*0).astype(np.uint8)
fine_contour_map = cv2.drawContours(fine_contour_map, contours, 0, (255), 8)
fine_contour, fine_h = cv2.findContours(fine_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
map_visu = cv2.drawContours(map_visu, fine_contour,0, (0,0,255),1)
# COMPUTE HARD-SPAWN AREA
hardspawn_contour_map = (map_.copy()*0).astype(np.uint8)
hardspawn_contour_map = cv2.drawContours(hardspawn_contour_map, fine_contour, 0, (255), 110)
hardspawn_contour, fine_h = cv2.findContours(hardspawn_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
map_visu = cv2.drawContours(map_visu, hardspawn_contour,1, (235,52,210),40)
# COMPUTE PERFECT NAVIGATION DISTANCE
distance_contour_map = (map_.copy()*0).astype(np.uint8)
ditance_contour_map = cv2.drawContours(distance_contour_map, fine_contour, 0, (255), 180)
distance_contour, fine_h = cv2.findContours(distance_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
map_visu = cv2.drawContours(map_visu, distance_contour,1, (255,204,0),30)
# MAKE SPAWN MAP
spawn_area = (map_.copy()*0).astype(np.uint8)
spawn_area = cv2.drawContours(spawn_area, hardspawn_contour ,1, (255), 40)
spawn_area = cv2.drawContours(spawn_area, distance_contour, 1, (255), 30)
# COMPUTE DISTANCE OF ALL PIXELS TO THE PERFECT NAVIGATION LINE
map_optimal_nav = np.ones_like(map_,dtype=np.uint8)*255
map_optimal_nav = cv2.drawContours(map_optimal_nav, distance_contour,1, (0),1)
map_dist2optimal_nav = cv2.distanceTransform(map_optimal_nav, cv2.DIST_L2, 5)
# MOVE TO CORRECT PROJECTION
spawn_area = cv2.rotate(spawn_area, cv2.ROTATE_90_COUNTERCLOCKWISE)
map_dist2optimal_nav = cv2.rotate(map_dist2optimal_nav, cv2.ROTATE_90_COUNTERCLOCKWISE)
spawn_area = cv2.flip(spawn_area, 0)
map_dist2optimal_nav = cv2.flip(map_dist2optimal_nav, 0)
good_contour = fine_contour[0]
# TAKES NAVIGATION LINE AND FITS SMOOTH SPLNE
x = distance_contour[1][:,0, 0]
y = distance_contour[1][:,0, 1]
print(x,y)
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.001, 0.001) 
out = interpolate.splev(unew, tck)
sx = out[0]
sy = out[1]
print(sx,sy)
error = 1
t = np.arange(sx.shape[0])
std = error * np.ones_like(t)
t2 = np.arange(sx.shape[0]*4)/4
fx = UnivariateSpline(t, sx, k=4, w=1 / np.sqrt(std))
fy = UnivariateSpline(t, sy, k=4, w=1 / np.sqrt(std))
# COMPUTE CURVATURE FROM SPLINE
sx2 = fx(t2)
sy2 = fy(t2)
print(sx2.shape,sy2.shape)
x1 = fx.derivative(1)(t2)
x2 = fx.derivative(2)(t2)
y1 = fy.derivative(1)(t2)
y2 = fy.derivative(2)(t2)
curvature = (x1* y2 - y1* x2) / np.power(x1** 2 + y1** 2, 1.5)
# COMPUTE RUNNING CURVATURE
max_speed = 1.5 #ms
ep_length = 60.0 #seconds
lake_length = 1400.0 #meters
print(sx2.shape,sy2.shape)
print(sx2.shape[0],sy2.shape[0])
window_size = int(0.25*max_speed*ep_length / (lake_length/sx2.shape[0]))
running_curvature = np.zeros_like(curvature)
for i in range(sx2.shape[0]):
    if i < sx2.shape[0] - window_size:
        running_curvature[i] = np.mean(np.abs(curvature[i:i+window_size]))
    else:
        running_curvature[i] = np.mean(np.abs((list(curvature[i:curvature.shape[0]])+list(curvature[0:i - curvature.shape[0] + window_size]))))
# DISCTRETIZE THE SHORE LINE
x_shore = fine_contour[0][:,0,0]
y_shore = fine_contour[0][:,0,1]
tck, u = interpolate.splprep([x_shore, y_shore], s=0)
unew = np.arange(0, 1.0001, 0.0001) 
out_shore = interpolate.splev(unew, tck)
sx_shore = out_shore[0]
sy_shore = out_shore[1]
# APPLY POLYNOMIAL FILTER
fsx_shore = savgol_filter(sx_shore,401,2)
fsy_shore = savgol_filter(sy_shore,401,2)
# COMPUTE DISTANCE (TO CHANGE BASED ON CEDRIC'S FEEDBACK)
diff_shore = (sx_shore - fsx_shore)**2 + (sy_shore-fsy_shore)**2
# WINDOWED STANDARD DEVIATION
diff_window = np.zeros_like(diff_shore)
for i in range(sx_shore.shape[0]):
    if (i > 50) and (sx_shore.shape[0] > i+50):
        diff_window[i] = np.std(diff_shore[i-50:i+50])
    elif i < 50: 
        diff_window[i] = np.std(list(diff_shore[0:i+50])+list(diff_shore[-(sx_shore.shape[0]-i+50):]))
    else:
        diff_window[i] = np.std(list(diff_shore[-(i-50):])+list(diff_shore[:sx_shore.shape[0]-i+50]))
rz = np.arctan2(y1,x1)
nav_line_pose = np.vstack((sx2, sy2, rz))
running_curvature_pose = np.vstack((sx2, sy2, running_curvature))
p = np.ones((101))/101
offset = [-1000,-3000]
spawn_poses = np.argwhere(spawn_area[:,:] == 255)

running_curvature_pose.shape
hard_spawn_cost = map_dist2optimal_nav
hard_spawn_poses = spawn_poses
hard_spawn_array = hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]]
print(hard_spawn_cost.shape)
print(hard_spawn_poses.shape)
print(hard_spawn_array.shape)
def create_weight_matrix(current_array):
    '''
    THIS FUNCTION RETURNS THE WEIGHT MATRIX OF THE CURVATURE MATRIX, IN ORDER TO HAVE A UNIFORM DISTRIBUTION OF THESE SAMPLES
    curvature: a numpy of m elements of size 3: x,y,curvature
    '''
    length_array = len(current_array)
    # Weight the samples
    hist_array = np.histogram(current_array, bins=101)
    sum_hist = np.array(hist_array[0])
    values_hist = np.array(hist_array[1])
    weight_matrix = np.zeros(length_array)
    for idx in range(length_array):
        current_curv = current_array[idx]
        idx_hist = np.where((values_hist-current_curv >= 0) == True)[0][0] - 1
        weight_matrix[idx] = 1/float(sum_hist[idx_hist])
    return weight_matrix
def sample_hard_spawn_fn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, sampled_pose=None, dict_diff_idx=None,
                         pdf="gaussian", difficulty=0, constant=0.1, do_weight=False, distance_threshold=200):
    ''' THIS FUNCTION SAMPLES A HARD SPAWNING POSITION, IF NO sampled_pose IS PROVIDED,
    THE POSITION IS PICKED AT RANDOM. OTHER_WISE IT WILL BE CLOSE FROM THE REQUESTED
    POSITION
    p_hardspawn: the probabilty of sampling a hard_spawn position given it's distance to the optimal trajectory. A vector of size 101.
    hard_spawn: a list of possible hard_spawn location of size 2: x,y
    hard_spawn_cost: a matrix of cost where the value in each pixel is the distance to the optimal navigation trajectory
    sanpled_pose: a list of 2 values: x and y
    '''
    if sampled_pose is not None:
        possibilities = hard_spawn_poses[np.argwhere(np.sqrt((hard_spawn_poses[:,0] - sampled_pose[0])**2+                                                              (hard_spawn_poses[:,1] - sampled_pose[1])**2) < distance_threshold)][:,0,:]
        hard_spawn_array = hard_spawn_cost[possibilities[:,0], possibilities[:,1]] 
    else:
        hard_spawn_array = hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]]
    length_array = len(hard_spawn_array)
    for idx in range(len(hard_spawn_array)):
        key = hard_spawn_array[idx]
        if key in dict_diff_idx.keys():
            dict_diff_idx[key].append(idx)
        else:
            dict_diff_idx[key] = [idx]
    dmax = np.max(hard_spawn_array)
    dmin = np.min(hard_spawn_array)
    mu = dmin + difficulty*(dmax-dmin)
    sigma = 0.1*(dmax-dmin) # 10% of the curvature range
    # Determine the importance of each curvature value
    if pdf == "gaussian": 
        prob_difficulty = gaussian(hard_spawn_array, mu, sigma)
    elif pdf == "constant-gaussian":
        prob_difficulty = [constant_gaussian(x, mu, sigma, constant=constant) for x in hard_spawn_array]
    elif pdf == "2sigma-gaussian":
        sigma_left = 4*sigma
        sigma_right = sigma
        prob_difficulty = [twosigma_gaussian(x, mu, sig_l=sigma_left, sig_r=sigma_right) for x in hard_spawn_array]
    else:
        raise Exception('Please provide a supported probability density function.')
    # Weight the probabilities
    if do_weight:
        weight_matrix = create_weight_matrix(hard_spawn_array)
        prob_difficulty = np.multiply(prob_difficulty, weight_matrix)
    # Sum of the array equals to 1
    prob_sum = np.sum(prob_difficulty)
    prob_difficulty = prob_difficulty/prob_sum
    target_diff = np.random.choice(hard_spawn_array, p=prob_difficulty)
    target_idx = np.random.choice(dict_diff_idx[target_diff])
    
    print("sampled_pose:", sampled_pose)
    print("difficulty:", difficulty)
    print("len hard_spawn_array:", len(hard_spawn_array))
    print("difficulty", target_idx, target_diff)
    if sampled_pose is not None:
        pose = possibilities[target_idx]
    else:
        pose = hard_spawn_poses[target_idx]
    print("pose", pose)
    return pose

if __name__ == '__main__':
    pdf = "gaussian"
    do_weight = True
    difficulty = 0.01
    sampled_pose = np.array([2306.90099185, 1434.08119941])
    sample_hard_spawn_fn(None, hard_spawn_poses, hard_spawn_cost, sampled_pose=sampled_pose, dict_diff_idx={},
                         pdf=pdf, difficulty=difficulty, do_weight=do_weight)
