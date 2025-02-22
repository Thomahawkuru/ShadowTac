import cv2
import numpy as np
from scipy.fft import ifftshift, fft2, ifft2

class PhotoStereoHeight:

    def __init__(self, numimg, angle, order, heightLight, sampling, shape=(600, 600), display=False):
        self.nbimg = numimg
        self.display = display
        self.angle = angle
        self.heightLight = heightLight
        self.sampling = sampling
        self.factorz = self.sampling * 1/30 *3
        self.shape = shape

        self.normalmap = []
        self.pgrads = []
        self.qgrads = []
        self.height = []
        self.backheight = np.zeros((np.array(shape[:2])/sampling).astype(int))

        self.max_height = 0

        self.calculate_light(order)

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

        self.light_mat = lightmat

    def createimagearray(self, image):
        
        width, height = self.shape[:2]

        image = cv2.resize(image, (int(height//self.sampling), int(width//self.sampling)))

        blue_channel, green_channel, red_channel = cv2.split(image)

        self.image_array = [blue_channel, green_channel, red_channel]

    def computenormalmap(self):

        self.normalmap = []
        self.pgrads = []
        self.qgrads = []
        input_array = self.image_array

        # Convert input array to float img array
        input_arr_conv = []
        for id in range (0, self.nbimg):
            im_fl = np.float32(input_array[id])
            im_fl = im_fl / 255
            input_arr_conv.append(im_fl)
        h = input_arr_conv[0].shape[0]
        w = input_arr_conv[0].shape[1]
        self.normalmap = np.zeros((h, w, 3), dtype=np.float32)
        self.pgrads = np.zeros((h, w), dtype=np.float32)
        self.qgrads = np.zeros((h, w), dtype=np.float32)
        lpinv = np.linalg.pinv(self.light_mat)
        intensities = []
        for imid in range(0, self.nbimg):
            a = np.array(input_arr_conv[imid]).reshape(-1)
            intensities.append(a)
            
        intensities = np.array(intensities)
        rho = np.einsum('ij,jk->ik', lpinv, intensities)
        self.normalmap[:, :, 0] = np.reshape(rho[0], (h, w))
        self.normalmap[:, :, 1] = np.reshape(rho[1], (h, w))
        self.normalmap[:, :, 2] = np.reshape(rho[2], (h, w))
        self.pgrads = self.normalmap[:, :, 0] / self.normalmap[:, :, 2]
        self.qgrads = self.normalmap[:, :, 1] / self.normalmap[:, :, 2]
        # self.normalmap = cv2.cvtColor(self.normalmap, cv2.COLOR_BGR2RGB)

    def computeheightmap(self):
        h = self.normalmap.shape[0]
        w = self.normalmap.shape[1]
        Z = np.zeros((h, w), dtype=np.float32)
        dfdxfft = fft2(self.pgrads)
        dfdyfft = fft2(self.qgrads)
        i = complex(0, 1)
        [nrows, ncols] = self.pgrads.shape
        [wx, wy] = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, ncols), np.linspace(np.pi/2, -np.pi/2, nrows))
        wx = ifftshift(wx)
        wy = ifftshift(wy)

        depthsfft = (i*wx*dfdxfft + i*wy*dfdyfft)/(wx**2 + wy**2 + 1e-10)

        Z = ifft2(depthsfft)
        self.height = (Z).real

        if self.display:
            cv2.imshow('z_norm', self.height)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def scaleheight(self):
        height_scaled = self.height - self.backheight
        height_scaled[height_scaled < 0] = 0
        height_scaled = height_scaled * self.factorz
        self.height_scaled = height_scaled
        self.height_mm_resize = cv2.resize(self.height_scaled, (self.shape[:2]))

    def draw_height(self, Black_Image):
        height_scaled_img = (self.height_mm_resize*255/10)
        height_scaled_img[height_scaled_img > 255] = 255
        height_scaled_img = height_scaled_img.astype(np.uint8)
        height_color = cv2.applyColorMap(height_scaled_img, cv2.COLORMAP_OCEAN)
        
        Black_Image = cv2.add(Black_Image, height_color)
        self.height_scaled_img = height_scaled_img
        return Black_Image
    
    def reset_height(self):
        self.backheight = self.height
        self.backheight = cv2.GaussianBlur(self.backheight,(55,55),0)
    
    def normal_tracking(self, img):
        self.createimagearray(img)
        self.computenormalmap()
        self.computeheightmap()
        self.scaleheight()
        self.max_height = np.max(self.height_scaled)
