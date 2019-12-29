
from PIL import Image
import glob, os

# Setting the points for cropped image

for infile in sorted(glob.glob('*.png')):
    file, ext = os.path.splitext(infile)

    im = Image.open(infile)

    width, height = im.size

    img_num = 0
    dim_x = 0
    dim_y = 0
    i = 0

    #column crop
    while i <= 8:
        left = dim_x
        top = dim_y
        right = left + 130
        bottom = top + 130

        #move down a row
        if i == 8 :
            dim_y += 130
            dim_x = 0 
            i = 0
            continue

        print (i)
        im1 = im.crop((left, top, right, bottom))
        im1.save('%d'%img_num + "crop.JPG", "JPEG")

        dim_x += 130
        i += 1
        img_num += 1


        if dim_y >= 2732 :
            break




# Shows the image in image viewer
#im1.show()
