
from PIL import Image
import glob, os

# Setting the points for cropped image



for infile in sorted(glob.glob('*.jpg')):
    file, ext = os.path.splitext(infile)

    im = Image.open(infile)

    width, height = im.size
    left = width/3
    top = height/5
    right = 2 * width/3
    bottom = 2 * height/5
    
    im1 = im.crop((left, top, right, bottom))
    im1.save(file + "crop.JPG", "JPEG")

# Shows the image in image viewer
#im1.show()
