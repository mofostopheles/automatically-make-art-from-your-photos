# -*- coding: utf8 -*-
'''
	USAGE
	❇️ Set your paths to input and output folders below.
	❇️ Default image source types are .jpg, but other types are supported.
	❇️ Update `image_path` to change image source types.
	❇️ Run script. Images will be processed and rendered to output directory.

	LICENSE
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Lesser General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Lesser General Public License for more details.
	You should have received a copy of the GNU Lesser General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
__author__ = "Arlo Emerson <arloemerson@gmail.com>"
__version__ = "11.0"
__date__ = "11/27/2020"

from PIL import Image, ImageOps, ImageFont, ImageDraw, ImageFilter, ImageChops
from cv2 import xphoto as xp
import numpy as np
import cv2, time, datetime, random, glob

class WatercolorPainter():

	def __init__(self):
		print('🐯 ' + self.__class__.__name__)

	def main(self):

		# ======================================================================
		# user defined settings
		# ======================================================================
		self.iteration_number = 26 # all images can be tracked back to this version number
		
		# set file input and output paths
		self.save_path_prefix = '../output/_' + str(self.iteration_number) + '_'
		self.image_path = '../input/*.jpg'
		self.signature_image_path = './signature_2020.jpg'

		# presets of edge and paint settings
		self.list_of_edge_maxxes = [10, 75, 400]
		self.list_of_paint_settings = [[100,2], [50,2]]
	
		# experimental presets:
		# list_of_paint_settings = [ [5,3], [10,2], [15,2], [15,6], [20,3], [30,2], [40,4], [50,2], [75,2], [100,2] ]
		# list_of_paint_settings = [[10,2], [15,2], [50,2], [75,2], [100,2], [200,2] ]
		# list_of_edge_maxxes = [0, 22, 50, 75, 100, 200, 220, 240, 250, 300, 400, 500, 600]
		# list_of_edge_maxxes = [22, 0, 10, 50, 75, 100, 200, 600]
		# ======================================================================
		
		self.timestamp = str( datetime.date.today() ) + '-' + str(time.time())
		self.image_dimension_w = 2000 # some default, will be reset
		self.image_dimension_h = 2000
		self.signature_image = Image.open( self.signature_image_path )
		self.signature_image.resize(( round(self.signature_image.width/2),round(self.signature_image.height/2)))

		# use this list of chars to generate interesting file names
		char_list = ["a", "i", "u", "o", "ka", "ki", "ku", "ke", "ko", "ni", "ga", "gi", "gu", "ge", "go", "sa", "shi", "su", "sa", "so", "za", "ji", "zu", "sa", "zo", "ta", "chi", "tsu", "te", "to", "da", "gi", "zu", "de", "do", "na", "ni", "nu", "ne", "no", "ha", "hi", "fu", "he", "ho"]
		self.j_name = ""
		
		# glob the images
		image_list = sorted(glob.glob( self.image_path ))

		for image in image_list:
			self.j_name = ""

			for x in range(0,3):
				self.j_name += char_list[ random.randrange(0, len(char_list)) ]

			load_image = Image.open( image )

			self.image_dimension_w = load_image.width
			self.image_dimension_h = load_image.height

			load_image = load_image.resize((self.image_dimension_w,self.image_dimension_h))
			new_image = Image.new('RGB', (self.image_dimension_w,self.image_dimension_h))
			new_image.paste(load_image, (0, 0))

			equalized = ImageOps.equalize(new_image)
			print("👨‍🎨 I'm saving an equalized version...")
			equalized.save( self.save_path_prefix + self.j_name + "_eq_" + self.timestamp + ".png" )

			self.make_edges(new_image)

		print("🏁 All done. Have a nice day.")

	def make_edges(self, new_image):

		# edges on contrast adjusted image
		alpha = 0.8
		beta = 30
		new_image = cv2.convertScaleAbs( np.asarray(new_image), alpha=alpha, beta=beta)

		# make a painting based on each edge
		for m in self.list_of_edge_maxxes:
			edges_variation = self.convert_to_cv2_edges3(new_image, m)
			self.make_painting(Image.fromarray(new_image), edges_variation, m)

	def make_painting(self, new_image, edges_variation, edge_index):

		# ======================================================================
		# create edge variations
		# ======================================================================
		paint_1 = xp.oilPainting( np.asarray( new_image ), 20, 2, cv2.COLOR_BGR2Lab)
		paint_1 = Image.fromarray(paint_1)
		edges_paint_1 = self.convert_to_cv2_edges(paint_1)

		# save a painting with edges
		file_name = self.save_path_prefix + self.j_name + "_study_edge_" + self.timestamp + ".png"
		print("👨‍🎨 I'm saving a painting with edges " + file_name)
		ImageOps.invert(edges_paint_1).save( file_name )

		# make a painting variation, note we are being called from another loop
		for paint_settings in self.list_of_paint_settings:
			self.make_paint_variation(new_image, edges_variation, edge_index, paint_1, edges_paint_1, paint_settings)

	def make_paint_variation(self, new_image, edges_variation, edge_index, paint_1, edges_paint_1, paint_settings):
		
		# used in later experiment
		paint_distance = paint_settings[0]
		paint_dynamic_range = paint_settings[1]
		paint_2 = xp.oilPainting( np.asarray( new_image ), paint_distance, paint_dynamic_range, cv2.COLOR_BGR2Lab)
		paint_2_edges = self.convert_to_cv2_edges(paint_2)
		paint_settings_short_name = "_" + str(paint_distance) + "-" + str(paint_dynamic_range) + "_"

		# ======================================================================
		# enhancements
		# ======================================================================
		edges_3 = self.image_offsetter(edges_variation, (1,1))
		doubled_edges_variation = ImageChops.screen(edges_variation, edges_3)
		composite_edges = ImageChops.screen(edges_paint_1, doubled_edges_variation)
		composite_edges = ImageOps.invert(composite_edges)
		find_edges = new_image.filter(ImageFilter.FIND_EDGES)
		find_edges = ImageOps.invert(find_edges)
		find_edges = ImageOps.grayscale(find_edges)
		find_edges = ImageOps.autocontrast(find_edges, cutoff=5, ignore=None)
		composite_edges_2 = ImageChops.screen(self.image_offsetter(paint_2_edges, (1,1)), paint_2_edges)
		composite_edges_2 = Image.fromarray( np.dstack([  np.asarray(composite_edges_2)  ]*3) )
		composite_edges_2 = ImageOps.invert(composite_edges_2)

		# composite of all edge passes
		all_edges = ImageChops.multiply(find_edges, composite_edges)

		# ======================================================================
		# save equalized, paintings, edges/outlines
		# ======================================================================
		rgb_edges = Image.fromarray( np.dstack([all_edges]*3) ) # Make it 3 channel, convert to pil
		doubled_edges_variation = Image.fromarray( np.dstack([  np.asarray(doubled_edges_variation)  ]*3) )
		doubled_edges_variation = ImageOps.invert(doubled_edges_variation)

		composite_edges_3 = ImageChops.multiply(doubled_edges_variation, composite_edges_2)
		# if edge_index == 400:
		triple_edge_vari = self.image_offsetter(doubled_edges_variation, (1,1))
		triple_edge_vari = self.image_offsetter(triple_edge_vari, (1,1))
		# triple_edge_vari = ImageChops.screen(edge_image1, edge_image2)
		file_name = self.save_path_prefix + self.j_name + "_outline_" + str(edge_index) + '_' + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a triple edge variation " + file_name)
		triple_edge_vari.save( file_name )
		# convert to RGB to prevent "images don't match" error when multiplying BW * RGB 
		triple_edge_vari = Image.fromarray( np.dstack([triple_edge_vari]*3) ) 

		# these two are the basic
		# paint 1 is a contrasty paint with tight outline
		paint_with_edges1 = ImageChops.multiply( self.filter_3(paint_1), rgb_edges)
		paint_with_edges1 = self.composite_signature(paint_with_edges1)
		file_name = self.save_path_prefix + self.j_name + "_study1_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges1.save( file_name)

		# paint 2 is contrasty with variation in the paint borders, but using the tight ouline
		paint_with_edges2 = ImageChops.multiply( self.filter_3(paint_2), rgb_edges)
		paint_with_edges2 = self.composite_signature(paint_with_edges2)
		file_name = self.save_path_prefix + self.j_name + "_study2_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges2.save( file_name)

		# paint 3 is uses the outline in the paint variation multiplied to the tight outline
		# this is two paint paintings overlaid
		painted_edge = Image.fromarray( np.dstack([composite_edges]*3) )
		paint_with_edges2 = ImageChops.multiply( self.filter_3(paint_2), painted_edge)
		paint_with_edges2 = self.composite_signature(paint_with_edges2)
		file_name = self.save_path_prefix + self.j_name + "_study3_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges2.save( file_name )

		# just add 1 and 2
		paint_with_edges3 = ImageChops.add( paint_with_edges1, paint_with_edges2, scale=2.0 )
		paint_with_edges3 = self.composite_signature(paint_with_edges3)
		file_name = self.save_path_prefix + self.j_name + "_study4_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges3.save( file_name)

		# many other variations to try below
		# multiply the paint edges to the previous
		paint_with_edges4 = ImageChops.multiply( Image.fromarray(paint_2), ImageOps.invert( Image.fromarray( np.dstack([paint_2_edges]*3) )) )
		paint_with_edges4 = self.composite_signature(paint_with_edges4)
		file_name = self.save_path_prefix + self.j_name + "_study5_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges4.save( file_name )

		paint_with_edges5 = ImageChops.add( paint_with_edges4, paint_with_edges2, scale=2.0 )
		paint_with_edges5 = ImageChops.multiply( paint_with_edges4, paint_with_edges2)
		paint_with_edges5 = self.composite_signature(paint_with_edges5)
		file_name = self.save_path_prefix + self.j_name + "_study6_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges5.save( file_name )

		paint_with_edges5 = ImageChops.blend( paint_with_edges4, paint_with_edges1, 0.3  )
		paint_with_edges6 = ImageChops.multiply( paint_with_edges5, triple_edge_vari)
		file_name = self.save_path_prefix + self.j_name + "_study7_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges6.save( file_name )

		paint_with_edges3 = ImageChops.add( paint_with_edges1, paint_with_edges2 )
		file_name = self.save_path_prefix + self.j_name + "_study8_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges3.save( file_name )

		paint_with_edges = ImageChops.multiply(Image.fromarray(paint_2), rgb_edges)
		file_name = self.save_path_prefix + self.j_name + "_study8_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_with_edges.save( file_name )

		washed_out_paint = ImageChops.multiply(  self.filter_3(paint_2), rgb_edges)
		file_name = self.save_path_prefix + self.j_name + "_study10_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		washed_out_paint.save( file_name )

		simple_edges = ImageChops.multiply( self.filter_3(paint_1), doubled_edges_variation)
		file_name = self.save_path_prefix + self.j_name + "_study11_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		simple_edges.save( file_name )

		simple_edges = ImageChops.multiply(  self.filter_3(paint_2), doubled_edges_variation)
		file_name = self.save_path_prefix + self.j_name + "_study12_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		simple_edges.save( file_name )

		paint_2_edges_painting = ImageChops.multiply(  self.filter_2(paint_2), composite_edges_2)
		file_name = self.save_path_prefix + self.j_name + "_study13_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_2_edges_painting.save( file_name )

		paint_2_edges_painting = ImageChops.multiply(  self.filter_2(paint_2), composite_edges_3)
		file_name = self.save_path_prefix + self.j_name + "_study14_" + str(edge_index) + paint_settings_short_name + self.timestamp + ".jpg"
		print("👨‍🎨 I'm saving a composition study " + file_name)
		paint_2_edges_painting.save( file_name )

		file_name = self.save_path_prefix + self.j_name + "_all_edges_" + str(edge_index) + "_" + self.timestamp + ".png"
		print("👨‍🎨 I'm saving an 'all edges' version " + file_name)
		all_edges.save( file_name )

		# ======================================================================
		# the end
		# ======================================================================

	def composite_signature(self, image):

		new_image = Image.new('RGB', ((image.width,image.height)), (255, 255, 255))
		new_image.paste(self.signature_image, (image.width - self.signature_image.width, image.height - self.signature_image.height))
		new_image = ImageChops.multiply( image, new_image  )
		return new_image

	def filter_1(self, image):

		brightness = 50
		contrast = 30
		img = np.int16( np.asarray(image))
		img = img * (contrast/127+1) - contrast + brightness
		img = np.clip(img, 0, 255)
		img = np.uint8(img)
		return Image.fromarray(img)

	def filter_2(self, image):

		brightness = 70
		contrast = 30
		img = np.int16( np.asarray(image))
		img = img * (contrast/127+1) - contrast + brightness
		img = np.clip(img, 0, 255)
		img = np.uint8(img)
		return Image.fromarray(img)

	def filter_3(self, image):

		brightness = 40
		contrast = 80
		img = np.int16( np.asarray(image))
		img = img * (contrast/127+1) - contrast + brightness
		img = np.clip(img, 0, 255)
		img = np.uint8(img)
		return Image.fromarray(img)

	def image_offsetter(self, image, tuple_offset=(1,1)):

		new_image = Image.new('RGB', ((self.image_dimension_w,self.image_dimension_h)))
		new_image.paste(image, tuple_offset)
		return ImageOps.grayscale(new_image)

	# returns a regular pil image
	def convert_to_cv2_edges(self, new_image):

		cv2_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(cv2_image, (3, 3), 0)
		auto = self.auto_canny(blurred)
		converted_image = Image.fromarray(auto)
		return converted_image

	def convert_to_cv2_edges2(self, new_image):

		cv2_image = np.asarray(new_image)
		blurred = cv2.GaussianBlur(cv2_image, (3, 3), 0)
		auto = self.auto_canny(blurred)
		converted_image = Image.fromarray(auto)
		return converted_image


	def convert_to_cv2_edges3(self, new_image, max_level):

		cv2_image = np.asarray(new_image)
		blurred = cv2.GaussianBlur(cv2_image, (3, 3), 0)
		auto = self.auto_canny2(blurred, max_level)
		converted_image = Image.fromarray(auto)
		return converted_image

	def technique_for_replacing_color(self):

		foo_image = np.zeros(((self.image_dimension_w,self.image_dimension_h),3), dtype="uint8")
		test_image = np.asarray(new_image)

		# for x in range(200,250):

		foo_image[np.where((test_image==[10,10,10]).all(axis=2))] = [255,250,0]
		foo_image = Image.fromarray(foo_image)
		x='test'
		foo_image.save('../color_replace_' + str(x) + '_x.png')

	def get_r_channel(self, image):

		data = image.getdata()
		r = [(d[0], 0, 0) for d in data]
		img = image
		img.putdata(r)
		return img

	def rgb_channels(self):

		# experimental
		data = new_image.getdata()
		# Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
		r = [(d[0], 0, 0) for d in data]
		g = [(0, d[1], 0) for d in data]
		b = [(0, 0, d[2]) for d in data]
		# img = new_image
		# img.putdata(r)
		# img.save('../output/r.png')
		# img.putdata(g)
		# img.save('../output/g.png')
		# img.putdata(b)
		# img.save('../output/b.png')

	# this method thanks to
	# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
	def auto_canny(self, image, sigma=0.33):

		# compute the median of the single channel pixel intensities
		v = np.median(image)
		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)
		return edged

	def auto_canny2(self, image, max_level, sigma=0.33 ):

		# compute the median of the single channel pixel intensities
		v = np.median(image)
		# apply automatic Canny edge detection using the computed median
		lower = int(max(max_level, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)
		return edged

# instantiate and run
runner = WatercolorPainter()
runner.main()
