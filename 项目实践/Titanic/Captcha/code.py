#!/usr/bin/env python
# encoding: utf-8

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from pylab import *
import random
import string

def get_random_char():
	return [random.choice(string.ascii_letters + string.digits) for _ in range(4)]

def get_random_color():
	# must return a tuple for color filling
	return (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))

def get_digit_pic():
	width = 320
	height = 100
	im = Image.new('RGB', (width, height), (180, 180, 180))

	fontstyle = "C:\\Users\\tylzh\\Downloads\\851.ttf"
	fontsize = int(min(im.size)/2)
	font = ImageFont.truetype(fontstyle, fontsize)

	# generate four random digits
	digits = get_random_char()

	# draw digits
	for i in range(4):
		ImageDraw.Draw(im).text(((i+1)*max(im.size)/6, im.size[1]/4), digits[i], font=font, fill=get_random_color())

	# add some noise
	for _ in range(random.randint(3000, 5000)):
		ImageDraw.Draw(im).point((random.randint(0, width), random.randint(0, height)), fill=get_random_color())

	# blur image
	im = im.filter(ImageFilter.BLUR)

	# save image
	im.save("".join(digits)+'.jpg', 'jpeg')

	figure()
	imshow(im)
	show()

if __name__ == '__main__':
	get_digit_pic()
