#from PIL import Image

#img = Image.open("logo.jpg")
#area = (562,43,1450,1072)
#cropped_img = img.crop(area)

#img.show()
#cropped_img.show()

#cropped_img.save("trans.jpg")

from PIL import Image
import socket
import sys
import os
import picamera
import time
from socket import error as SocketError

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   
host = "192.168.0.2"
port = 5030

#server connection
s.connect((host, port))


##### camera
#camera = picamera.PiCamera()
#camera.video_stabilization = True
#camera.capture('logo.jpg')
camera = picamera.PiCamera()
camera.hflip = True
camera.vflip = True
camera.sharpness = 100
camera.video_stabilization = True
camera.capture('logo.jpg')


img = Image.open("logo.jpg")
area = (532,43,1450,1072)
cropped_img = img.crop(area)

#img.show()
#cropped_img.show()

cropped_img.save("trans.jpg")

#image...
f = open('trans.jpg', 'rb')
#f = open('n02917067_16346.JPEG', 'rb')
l = f.read(1024)

while (1):
        #print 'Sending..'
        s.send(l)
        l = f.read(1024)
        if not l: break
       

f.close()
s.close()

