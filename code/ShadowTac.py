'''
Description:
This script defines the ShadowTac class, which contains all the functionalities of the sensor.
The following keyboard inputs are available: 

Commands:
    - 'h' : Start the height map processing pipeline.
    - 'r' : Reset the background height.
    - 'c' : Start the contact aera detection.
    - 't' : Start dimple tracking pipeline.
    - 's' : Save an image (each press saves an image). If this is the first press of 's', it starts video recording, which will continue until the program exits.
    - 'e' : Start all pipelines simultaneously.
    - 'q' : Quit the program.

Notes:
- All captured media are saved in the defined directory.
- Video recording begins with the first 's' keypress and stops when the program exits.

'''

import os
import time
import cv2
import numpy as np
import pypylon.pylon as py
import keyboard
from DimpleTrackingClass import DimpleTrack
from PhotostereoClass import PhotoStereoHeight
import cropImage as crop


class Shadowtac:

    def __init__(self, dir):

        #Init boolean flags for functionalities
        self.TrackPattern = False
        self.DetectContactArea = False
        self.Height = False
        self.Display2D = True
        self.Save_Video = True
        self.Crop_Image = True
        self.exitprogram = False
        self.contours = None
        self.start_time = 0
        self.freq = 0

        #Init directories
        self.base_dir = dir
        self.save_folder = 'save/'
        self.save_dir = self.base_dir + self.save_folder
        os.makedirs(self.save_dir, exist_ok=True)

        # Marker tracking param
        self.marker_radius = 15

        # Main Image param
        self.radius = 600
        self.sizedisp = 700

        # LED Param
        self.heightLight = 1
        self.angle = 45
        self.order = 'RBG'

        # Heght computation param
        self.samplingheight = 8

        # MarkerTrack Param
        self.path_ref = self.base_dir + 'ImgInitSensor.npy'
        
        # Save data param
        self.img_count = 0
        self.Start_video = False


    def initialize(self):
        # Initialize tactile sensors camera
        self.camera = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        self.camera.ExposureAuto.Value = "Off"
        self.camera.ExposureTime.Value = 2000.0
        self.camera.GainAuto.Value = "Off"
        self.camera.Gain.Value = 0

        self.converter = py.ImageFormatConverter()
        self.converter.OutputPixelFormat = py.PixelType_BGR8packed
        self.converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned


    def get_image(self):
        
        timeout = 2000
        self.start_time = time.time()

        while self.camera.IsGrabbing():
            self.freq = (1 / (time.time() - self.start_time))
            self.start_time = time.time()

            self.grabResult = self.camera.RetrieveResult(timeout, py.TimeoutHandling_ThrowException)
            
            if self.grabResult.GrabSucceeded():
                image = self.converter.Convert(self.grabResult)
                image = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3), dtype=np.uint8)
        
                if self.Crop_Image:
                    image = crop.crop_image_small(image, self.radius)
                    image = crop.crop_image(image, self.radius)

                self.width = image.shape[0]
                self.height = image.shape[1]

                self.grabResult.Release()
                image_display = image.copy()

                return image, image_display

            else:
                print("Error: ", self.grabResult.ErrorCode, self.grabResult.ErrorDescription)
                
            self.grabResult.Release()


    def disp_image(self, image, image_display):
        
        if self.Display2D:

            black_image = np.zeros_like(image)
            
            if self.TrackPattern:
                black_image = self.dtrack.draw_tracking(black_image)
            
            if self.Height:
                black_image = self.psh.draw_height(black_image)

            if self.DetectContactArea:
                black_image = cv2.drawContours(black_image, self.contours, -1, (255, 255, 255), 4)  # -1 means drawing all contours, (255, 255, 255) is white color, 1 is the thickness
      
            # Display result
            self.ratio = self.height/self.width
            
            image_display = cv2.flip(cv2.resize(image_display, (int(self.sizedisp*self.ratio), self.sizedisp)),0)
            black_image_display = cv2.flip(cv2.resize(black_image, (int(self.sizedisp*self.ratio), self.sizedisp)),0)

            new_width = 400
            new_height = 400

            resized_raw = cv2.resize(image_display, (new_width, new_height))
            resized_black = cv2.resize(black_image_display, (new_width, new_height))

            cv2.imshow('Raw Image', resized_raw)
            cv2.imshow('Disp. field', resized_black)

            
            cv2.waitKey(1)

        if self.Start_video:
            self.out.write(image)
            self.out_black.write(black_image)
        
        return black_image

        
    def check_keyboard_inputs(self, image, black_image):

        if self.Start_video:
            self.out.write(image)
            self.out_black.write(black_image)

        if keyboard.is_pressed('r'):
            if self.Height:
                self.psh.normal_tracking(image)
                self.psh.reset_height()
            # continue

        if keyboard.is_pressed('h'):
            self.Height = not self.Height
            print('Height:', self.Height)

            self.psh = PhotoStereoHeight(3, self.angle, self.order, self.heightLight, self.samplingheight, image.shape)
            self.psh.normal_tracking(image)
            self.psh.reset_height()
            
        if keyboard.is_pressed('t'):
            self.TrackPattern = not self.TrackPattern
            print('Track Pattern:', self.TrackPattern)

            self.dtrack = DimpleTrack(self.angle, self.order, self.heightLight, self.marker_radius, self.path_ref, image.shape)
            print('nb marker: ', self.dtrack.init_centroid.shape)

        if keyboard.is_pressed('c'):
            if self.Height:
                self.DetectContactArea = not self.DetectContactArea
                print('Detect Contact Area', self.DetectContactArea)
         
        if keyboard.is_pressed('e'):
            if self.TrackPattern and self.Height:
                self.TrackPattern = False
                self.Height = False
            else:
                self.TrackPattern = True
                self.Height = True  

            self.dtrack = DimpleTrack(self.angle, self.order, self.heightLight, self.marker_radius, self.path_ref, image.shape)
            print('nb marker: ', self.dtrack.init_centroid.shape)

            self.psh = PhotoStereoHeight(3, self.angle, self.order, self.heightLight, self.samplingheight, image.shape)
            self.psh.normal_tracking(image)
            self.psh.reset_height()

            # time.sleep(0.1)       

        if keyboard.is_pressed('s'):
            
            if self.Save_Video and not self.Start_video:
                # Define the codec and create VideoWriter object 
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                # out = cv2.VideoWriter(save_dir + 'output.avi', fourcc, 20.0, (1600, 1200))
                self.out = cv2.VideoWriter(self.save_dir + 'output.avi', self.fourcc, self.freq, (2*self.radius, 2*self.radius))
                self.out_black = cv2.VideoWriter(self.save_dir + 'output_black.avi', self.fourcc, self.freq, (black_image.shape[0], black_image.shape[1]))
                self.Start_video = True 

            self.image_save_path = self.save_dir + f'{self.img_count}_raw.tiff'
            cv2.imwrite(self.image_save_path, image)
            self.image_save_path = self.save_dir + f'{self.img_count}.tiff'
            cv2.imwrite(self.image_save_path, black_image)
            print(f'Image {self.img_count} succesfully saved at {self.save_dir}')
            if self.img_count == 0 and self.TrackPattern == True:
                self.dtrackSave = DimpleTrack(self.angle, self.order, self.heightLight, self.marker_radius, self.path_ref, image.shape)
                self.dtrackSave.init_centroid = None
                self.dtrackSave.IsolateMarkerHSV(image)
                self.dtrackSave.detect_marker()
                print('nb marker: ', self.dtrack.init_centroid.shape)
                np.save(self.save_dir + 'ImgInitSensor_new.npy', self.dtrackSave.init_centroid)
            self.img_count += 1    

        if keyboard.is_pressed('q'):
            self.grabResult.Release()
            self.camera.StopGrabbing()
            self.exitprogram = True
            print('Sensor closed.')
            time.sleep(0.1)


    def compute_3d_displacement(self):
    
        displacement = np.full((len(self.dtrack.points_init), 3), None) #old one, works with full disp and MLP+projection
        displacement[:,:2] = self.dtrack.point_position_mm - self.dtrack.points_init
        
        height_point = self.psh.height_mm_resize[self.dtrack.x_coords, self.dtrack.y_coords]
        displacement[:,2] = height_point    
        return displacement.astype(float).reshape(1, displacement.shape[0]*displacement.shape[1])


    def get_tactile_data(self, image):

        if self.TrackPattern:
            self.dtrack.lateral_tracking(image)

        if self.Height:
            self.psh.normal_tracking(image)

            if self.DetectContactArea:
                self.height_mm_blur = cv2.GaussianBlur(self.psh.height_mm_resize, (21, 21), 0)
                self.thresholded = (cv2.threshold(self.height_mm_blur, 1, 255, cv2.THRESH_BINARY)[1]).astype(np.uint8)
                self.contours, _ = cv2.findContours(self.thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        



'''
Run the sensor
'''
if __name__ == "__main__":

    dir = r'path/to/your/code/directory/'
    sensor = Shadowtac(dir)
    sensor.initialize()
    while True:
        img, disp_img = sensor.get_image()
        blk_img = sensor.disp_image(img, disp_img)
        sensor.get_tactile_data(img)
        sensor.check_keyboard_inputs(img, blk_img)

        #check working frequency
        sensor.freq = (1 / (time.time() - sensor.start_time))
        print('frequency:', ("%.2f" % sensor.freq))

        if sensor.exitprogram:
            print('Terminating program...')
            time.sleep(0.1)
            break