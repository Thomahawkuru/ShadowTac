import cv2
import numpy as np
from scipy.spatial import KDTree
import time

class DimpleTrack:

    def __init__(self, angle, order, heightLight, marker_radius, path_ref, shape=(600, 600)):

        self.angle = angle
        self.heightLight = heightLight

        self.init_centroid = None
        self.marker_radius = marker_radius
        self.min_centroid_size = (marker_radius/2)**2*np.pi*0.3
        self.max_distance = 3*self.marker_radius
        self.shape = shape

        self.factorxy = 35/1200
        self.factorthreshold = 0.85

        self.calculate_light(order)

        self.pos_init = np.load(path_ref, allow_pickle=True)
        self.load_marker()

        self.track_markers()
        self.scaleposition()
        self.points_init = self.point_position_mm

    def calculate_light(self, order):
        
        if order == 'RGB':
            order = -1
        elif order == 'RBG':
            order = 1
        else: order = 0

        lightmat = np.zeros((3,3))
        for i in range(3):
            angle_act = (self.angle + i*120*order)% (360)
            rad = (angle_act)/180*np.pi % (2*np.pi)
            lightmat[i,:] = [np.cos(rad), np.sin(rad), self.heightLight]
            # print(angle_act)

        self.light_mat = lightmat

    def roll_image(self, image):

        blue_channel, green_channel, red_channel = cv2.split(image)

        mult = int(-self.marker_radius)

        light_mat_int = (self.light_mat*mult).astype(int)

        n_blue_shift_hor = light_mat_int[0,0]
        n_blue_shift_vert = light_mat_int[0,1]

        n_green_shifted_hor = light_mat_int[1,0]
        n_green_shifted_vert = light_mat_int[1,1]

        n_red_shift_hor = light_mat_int[2,0]
        n_red_shift_vert = light_mat_int[2,1]

        # Shift each channel accordingly
        blue_shifted = np.roll(blue_channel, n_blue_shift_vert, axis=0)
        blue_shifted = np.roll(blue_shifted, -n_blue_shift_hor, axis=1)

        green_shifted = np.roll(green_channel, n_green_shifted_vert, axis=0)
        green_shifted = np.roll(green_shifted, -n_green_shifted_hor, axis=1)

        red_shifted = np.roll(red_channel, n_red_shift_vert, axis=0)
        red_shifted = np.roll(red_shifted, -n_red_shift_hor, axis=1)

        shifted_image_np = cv2.merge((blue_shifted, green_shifted, red_shifted))

        return shifted_image_np

    def IsolateMarkerHSV(self, image):

        start_time = time.time()
        image = cv2.blur(image, (11, 11))
        
        factor = 5

        window_blur = int((self.marker_radius * 2)/factor + 1)
        window_blur_large = int((self.marker_radius * 6)/factor + 1)

        shifted_image_np = self.roll_image(image)
        small_image = cv2.resize(shifted_image_np, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)

        hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
        v_channel = hsv_image[:, :, 2]

        gauss = cv2.blur(v_channel, (window_blur, window_blur))
        gauss_large = cv2.blur(gauss, (window_blur_large, window_blur_large))

        start_time = time.time()

        # average_value_float = -gauss.astype(float) + 2*gauss_large.astype(float) - 17
        average_value_float = (-gauss.astype(float) + 2*gauss_large.astype(float))*self.factorthreshold
        # average_value_float = gauss*0.85
        
        dark_spots = (v_channel < (average_value_float)).astype(np.uint8)*255

        cleaned_mask = dark_spots

        # cv2.imshow('dark_spots', cv2.resize(dark_spots, (800,800)))

        kernel_size = int(5/factor)
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        self.imgsegmented = cv2.resize(cleaned_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


        # cv2.imshow('cleaned_mask', cv2.resize(self.imgsegmented, (800,800)))

    def load_marker(self):

        self.init_centroid = self.pos_init
        self.prev_positions = self.init_centroid
        self.prev_displacement = np.zeros_like(self.init_centroid)

        self.kdtreeinit = KDTree(self.init_centroid)
        self.dist_init, self.idx_init = self.kdtreeinit.query(self.init_centroid, k=7)
        mask = np.std(self.dist_init[:,1:], axis=1) > np.mean(np.std(self.dist_init[:,1:], axis=1))
        self.idx_edge = self.idx_init[mask,0]
        self.edge_centroid = self.init_centroid[self.idx_edge]

        tracked_markers = np.full((len(self.prev_positions), 3, 2), None)
        tracked_markers[:,0,:] = self.prev_positions
        tracked_markers[:,1,:] = self.prev_positions
        self.tracked_markers = tracked_markers

        self.curr_centroid = self.init_centroid

    def detect_marker(self):

        # start_time = time.time()
        frame = self.imgsegmented
        # Ensure the frame is binary
        ret, binary_frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        
        # Perform connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame)
        frame_sizes = np.array(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label 0
            
        mask = (frame_sizes > self.min_centroid_size) & (frame_sizes < 300)
        filtered_centroids = centroids[1:][mask]
        filtered_centroids = np.array(filtered_centroids)

        if self.init_centroid is None:

            self.init_centroid = filtered_centroids
            self.prev_positions = self.init_centroid
            self.prev_displacement = np.zeros_like(self.init_centroid)

            self.kdtreeinit = KDTree(self.init_centroid)
            self.dist_init, self.idx_init = self.kdtreeinit.query(self.init_centroid, k=7)
            mask = np.std(self.dist_init[:,1:], axis=1) > np.mean(np.std(self.dist_init[:,1:], axis=1))
            self.idx_edge = self.idx_init[mask,0]
            self.edge_centroid = self.init_centroid[self.idx_edge]
        
            print('nb centroid init: ', len(self.init_centroid))

        self.curr_centroid = filtered_centroids
        # print('elasped time detecttttttttttttttttttttttt', time.time() - start_time)

    def track_markers(self):
        if len(self.curr_centroid) == 0 or len(self.prev_positions) == 0:
            return []
        start_time = time.time()
        kdtreeprev = KDTree(self.prev_positions)
        tracked_markers = np.full((len(self.prev_positions), 3, 2), None)
        tracked_markers[:,0,:] = self.prev_positions
        tracked_markers[:,1,:] = self.prev_positions

        valid_curr_positions = []
        displacement = self.prev_displacement
        idx_array_prev = []
        
        # Query all current positions at once
        distances, indices = kdtreeprev.query(self.curr_centroid, k=1)
        # Mask to filter points within max_distance
        mask = distances < self.max_distance
        idx_array_prev = indices[mask]
        valid_distances = distances[mask]
        valid_curr_positions = self.curr_centroid[mask]
        tracked_markers[idx_array_prev, 0, :] = self.prev_positions[idx_array_prev]
        tracked_markers[idx_array_prev, 1, :] = valid_curr_positions
        tracked_markers[idx_array_prev, 2, :] = [0, 0]
        
        displacement[idx_array_prev] = self.prev_positions[idx_array_prev] - valid_curr_positions

        # print('elasped time ', time.time() - start_time)
        start_time = time.time()

        unused_indices_prev = list(set(range(len(self.prev_positions))) - set(idx_array_prev))


        # unused_edge_indices = np.intersect1d(unused_indices_prev, self.idx_edge)
        # unused_center_indices = np.setdiff1d(unused_indices_prev, unused_edge_indices)

        if unused_indices_prev:
            # prev_pos_tmp = self.prev_positions[unused_center_indices]
            # distances, indices = self.dist_init[unused_center_indices, :], self.idx_init[unused_center_indices, :]
            # cur_pos_estimated = np.mean(self.prev_positions[indices], axis = 1)
            # tracked_markers[unused_center_indices, 0, :] = prev_pos_tmp
            # tracked_markers[unused_center_indices, 1, :] = cur_pos_estimated
            # tracked_markers[unused_center_indices, 2, :] = [1, 1]

            for idx in unused_indices_prev:
                distances, indices = self.dist_init[idx, :], self.idx_init[idx, :]
                try:
                    mask = distances[1:] < 7*self.marker_radius
                    close_centroid_idx = indices[1:][mask]
                    close_displacement = tracked_markers[close_centroid_idx,1,:] - self.prev_positions[close_centroid_idx]
                    new_edge_centroid = self.prev_positions[idx] + np.mean(close_displacement, axis=0)
                    tracked_markers[idx] = np.array([self.prev_positions[idx], new_edge_centroid, [2, 2]])
                except:
                    tracked_markers[idx] = np.array([self.prev_positions[idx], self.prev_positions[idx], [3, 3]])

        # print('elasped time ', time.time() - start_time)
        
        self.tracked_markers = np.array(tracked_markers)
        self.prev_positions = self.tracked_markers[:,1,:]
        self.prev_displacement = displacement

    def draw_tracking(self, frame):

        size_circle=15

        for i in range(len(self.tracked_markers)):
            prev_pos, curr_pos, value = self.tracked_markers[i]
            init_pos = self.init_centroid[i]
            if any(value == [0, 0]):
                cv2.circle(frame, tuple((curr_pos).astype(int)), size_circle, (0, 0, 255), -1)
            elif any(value == [1, 1]):
                cv2.circle(frame, tuple((curr_pos).astype(int)), size_circle, (0, 0, 255), -1)
            elif any(value == [2, 2]):
                cv2.circle(frame, tuple((curr_pos).astype(int)), size_circle, (0, 0, 255), -1)
            else:
                cv2.circle(frame, tuple((curr_pos).astype(int)), size_circle, (0, 0, 255), -1)
            # cv2.line(frame, tuple((prev_pos).astype(int)), tuple((curr_pos).astype(int)), (255, 255, 255), 1)
            cv2.line(frame, tuple((init_pos).astype(int)), tuple((curr_pos).astype(int)), (255, 255, 255), 5)

        # cv2.circle(frame, (200,200), 30, (255, 0, 0), -1)
        return frame

    def scaleposition(self):

        self.point_position_mm = self.tracked_markers[:,0,:]*self.factorxy

        point_position = self.tracked_markers[:,0,:]
        x_coords, y_coords = point_position[:,1].astype(int), point_position[:,0].astype(int)
        self.x_coords = np.clip(x_coords, 0, self.shape[0]-1)
        self.y_coords = np.clip(y_coords, 0, self.shape[1]-1)

    def lateral_tracking(self, img):
        self.IsolateMarkerHSV(img)
        self.detect_marker()
        self.track_markers()
        self.scaleposition()
