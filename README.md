# ShadowTac Overview
This repository provides step-by-step instructions to build and operate the ShadowTac vision based tactile sensor.

The ShadowTac sensor uses a miniature camera to capture high-resolution images of contact interactions with its soft tactile membrane. This membrane, coated with reflective paint, features a dense array of submillimeter dimples. When illuminated by three RBG LEDs, the membrane and dimples generate peculiar colored shadows that get captured by the camera. By processing these images, ShadowTac resolves a dense displacement field in both normal and tangential directions. This makes ShadowTac ideal for capturing dynamic interactions between a robotic fingertip and delicate objects.

ShadowTac is designed for easy and cost-effective manufacturing, requiring no specialized or expensive equipment. The dimple pattern can be optimized to minimize camera obstruction typical of the traditional opaque markers used for lateral tracking. Moreover, this pattern is formed directly through the molding process of the membrane, allowing seamless application on curved sensing surfaces, as opposed to opaque markers.

For further information, please read our paper: 

G. Vitrani*, B. Pasquale*, and M. Wiertlewski, "ShadowTac: dense measurement of shear and normal deformation of a
tactile membrane from colored shadows" [Accepted for ICRA25 - coming soon] 

*These authors contributed equally.

Please cite our work if you intend to use it (see additional resources below).



Here is a short video highlighting the key features of the sensor:


https://github.com/user-attachments/assets/36d0550c-10cc-456a-a98c-f12cc2762915



# Repository Content

```
ShadowTac/
│── 📁 CADs/                    # Contains files to print
│   ├── sensor_base.step        # Base to mount you sensor
│   ├── sensor_base.stl          
│   ├── sensor_casing.step      # Sensor casing
│   ├── sensor_casing.stl       
│   ├── silicone_mold.step      # Mold to cast tactile membrane
│   ├── silicone_mold.stl       
│   ├── silicone_mold_top.step  # Top part of the silicone mold
│   ├── silicone_mold_top.stl  
│── 📁 code/                    # Contains code to run the sensor
│   ├── cropImage.py            # Function to crop tactile image
│   ├── DimpleTrackingClass.py  # Class for lateral tracking of dimples
│   ├── PhotostereoClass.py     # Class for normal tracking
│   ├── ShadowTac.py            # Class to handle the sensor functionalities
│   ├── ImgInitSensor.npy       # Initial position of markers (Note: must be replaced with new sensor)
│   ├──📁 save/                 # Folder where images, videos and dimple disposition are saved 
│── 📁 other/                   # Other useful files
│   ├── rgb_led.ino             # Light LEDs
│── LICENSE                     # License file
│── README.md                   # Main README file

```

##


# Build Instructions

### Tactile membrane

![membrane manufacturing](https://github.com/user-attachments/assets/e48875b8-51e0-4144-bfe4-371e8c607c48)
Pictures illustrating the silicone fabrication process.

- **Silicone mold fabrication**: it must be 3D printed using an SLA printer for high dimple precision. The top part of the mold can be printed using a FDA printer. Both files (`sensor_casing.stl` and `sensor_casing_top.stl`) are available in the 📁 CADs folder. 

- **Silicone preparation**: the Smooth-On Solaris silicone is used, thinned with up to 30% Smooth-On Silicone Thinner for reduced shore hardness. The silicone compounds should be prepared as indicated on the vendor's website.

- **Degassing, casting and curing**: The silicone mixture must be degassed before pouring to eliminate air bubbles. A demolding agent should be applied on the mold to facilitate the removal of the cured silicone. Then, the silicone compound can be poured into the mold and left to cure for 24 hours. After this time, the silicone tactile membrane can be carefully demolded.

- **Paint preparation**: A paint mixture (Smooth-On Psychopaint A/B, NOVOCS Matte solvent, aluminum powder, White Silc Pig) must be prepared. The paint compounds should be prepared as indicated on the vendor's website.

- **Silicone painting**: The paint mixture must be poured uniformally over the silicone membrane. For an optimal result, the paint should completely penetrate each dimple. This can be achieved by manually spreading a small amount of paint over the dimples before pouring the remaining mixture. 

- **Bonding to PMMA**: The painted silicone can be bonded to a 3mm thick, 34mm diameter PMMA plate using Smooth-On SilPoxy adhesive. The surfaces must be cleaned before bonding, and small twisting and compressing motions help to remove air bubbles.

Below you can find two tables with silicone and paint compositions. Note that these compositions might be varied slightly without affecting the functionalities of the sensor. Feel free to experiment with them.

#### Silicone Composition

| Component             | Quantity (g) | Percentage |
|----------------------|-------------|------------|
| Silicone Solaris A   | 5           | 36%        |
| Silicone Solaris B   | 5           | 36%        |
| Silicone Thinner     | 4           | 28%        |

#### Paint Composition

| Component         | Quantity (g) | Percentage |
|------------------|-------------|------------|
| SilPoxy A       | 1.2         | 18%        |
| SilPoxy B       | 1.2         | 18%        |
| NOVOCS Matte    | 3.5         | 50%        |
| Aluminium Powder | 0.8         | 12.5%      |
| SilPig White    | 0.08        | 1.5%       |


### Sensor Casing and Illumination

- **Sensor casing**: 3D printed with FDA printer. The file `sensor_casing.stl` is available in the 📁CADs folder.
- **LED placement**: RGB LEDs are soldered and secured inside the casing using hot glue, ensuring they do not obstruct the emission cone.
- **Silicone-casing assembly**: The PMMA-silicone assembly is affixed to the sensor casing using standard super glue. 

### Camera Hardware
- **Camera**: Basler daA1920-160uc (S-Mount).
- **Lens**: Evetar M13B02820W.
- **Final assembly**: the camera can be assembled with the casing using four M2 screws and nuts.
##



# Running the Sensor

### Setup and Imports

Once the ShadowTac sensor is fully assembled, connect the camera to a computer by using a cable usb-micro b 3.0.

You can run the sensor by using the python codes provided in the 📁code folder of this repository.  

These codes import the following python libraties. Make sure you install the missing dependencies on your system.  

```python
import os  
import time  
import keyboard
import cv2  
import numpy as np  
from scipy.spatial import KDTree  
from scipy.fft import ifftshift, fft2, ifft2  
import pypylon.pylon as py  
```

After all the dependencies have been installed, open `ShadowTac.py`. This code defines the ShadowTac class, which contains all the functionalities of the ShadowTac. Among other functions, ShadowTac.py manages `DimpleTrackingClass.py` and `PhotostereoClass.py`, which handle the lateral and normal processing pipelines, respectively.

Before running the script, update the following line by specifying the full path to the directory where the 📁code folder is located on your system:

```python
dir =  r'path/to/your/code/directory/'
```

Once updated, you can run the sensor by executing this script directly, as it includes a main section at the bottom.

You can input the following keyboard commands to activate or deactivate sensor functionalities:

```
Commands:
    - 'h' : Start the height map processing pipeline.
    - 'r' : Reset the background height.
    - 'c' : Start the contact aera detection.
    - 't' : Start dimple tracking pipeline.
    - 's' : Save an image (each press saves an image). If this is the first press of 's', it starts video recording, which will continue until the program exits.
    - 'e' : Start all pipelines simultaneously.
    - 'q' : Quit the program.
```

### Sensor First Run
When a new sensor is assembled, you must save a new .npy file containing the initial dimple positions. This is essential because the lateral displacement pipeline relies on a predefined dimple layout to track movement from its initial state.

To do this, press the 's' key, which will generate a file named `ImgInitSensor_new.npy` in your 📁save folder. Rename this file to `ImgInitSensor.npy` and move it to the 📁code folder. Once completed, the sensor will correctly recognize the initial dimple positions, ensuring proper functionality of both the normal and tangential processing pipelines.


### Note
If you are using Ubuntu, the keyboard module may require admin privileges to function properly. To resolve this, add the following line to your launch.json configuration file:

```json
"sudo": true,
```

##

# Additional Resources


### Citation

For further information, please read our paper: 

G. Vitrani*, B. Pasquale*, and M. Wiertlewski, "ShadowTac: dense measurement of shear and normal deformation of a
tactile membrane from colored shadows" [Accepted for ICRA25 - coming soon] 

*These authors contributed equally.

Please cite our work if you intend to use it.

[Coming soon]


### Contacts
- **Giuseppe Vitrani**, Delft University of Technology, Cognitive Robotics Department, 2628 CD, Delft, The Netherlands - g.vitrani@tudelft.nl
- **Basile Pasquale**, École Polytechnique Fédérale de Lausanne, Rte Cantonale, 1015 Lausanne, Switzerland - basilepsqle6@gmail.com
- **Michaël Wiertlewski**, Delft University of Technology, Cognitive Robotics Department, 2628 CD, Delft, The Netherlands - m.wiertlewski@tudelft.nl


# License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
