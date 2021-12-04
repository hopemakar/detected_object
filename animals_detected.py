import os
import time
import PIL
from PIL import ImageFont, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import humanfriendly
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
from tqdm import tqdm


log_filename = os.path.join(os.getcwd(), 'uploader_logfile.txt')
log_file = open(log_filename, 'w')


DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DETECTION_FILENAME_INSERT = '_detections'
EMPTY_FILENAME_INSERT = "_empty"

DEFAULT_LINE_WIDTH = 10

bbox_categories = [
	{'id': 0, 'name': 'empty'},
	{'id': 1, 'name': 'animal'},
	{'id': 2, 'name': 'person'},
	{'id': 3, 'name': 'group'},
	{'id': 4, 'name': 'vehicle'}
]

bbox_category_str_id_to_name = {}

for cat in bbox_categories:
	bbox_category_str_id_to_name[str(cat['id'])] = cat['name']

def load_model(checkpoint):

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(checkpoint, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	return detection_graph


def generate_detections(detection_graph,images):

	if not isinstance(images,list):
		images = [images]
	else:
		images = images.copy()

	log_file.write('Loading images...' + '\n')
	start_time = time.time()

	for iImage,image in enumerate(tqdm(images)):
		if isinstance(image,str):
	        
			image = PIL.Image.open(image).convert("RGB")
			image = np.array(image)

			nChannels = image.shape[2]
			if nChannels > 3:
				log_file.write('Warning: trimming channels from image' + '\n')
				image = image[:,:,0:3]
			images[iImage] = image
		else:
			assert isinstance(image,np.ndarray)

	elapsed = time.time() - start_time
	log_file.write("Finished loading {} file(s) in {} \n".format(len(images),
		humanfriendly.format_timespan(elapsed)))    

	boxes = []
	scores = []
	classes = []

	n_images = len(images)

	log_file.write('Running detector...' + '\n')    
	start_time = time.time()
	first_image_complete_time = None

	with detection_graph.as_default():
	    
		with tf.compat.v1.Session(graph=detection_graph) as sess:

			for iImage,imageNP in tqdm(enumerate(images)): 
	            
				imageNP_expanded = np.expand_dims(imageNP, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				box = detection_graph.get_tensor_by_name('detection_boxes:0')
				score = detection_graph.get_tensor_by_name('detection_scores:0')
				clss = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')

				(box, score, clss, num_detections) = sess.run(
				        [box, score, clss, num_detections],
				        feed_dict={image_tensor: imageNP_expanded})

				boxes.append(box)
				scores.append(score)
				classes.append(clss)

				if iImage == 0:
					first_image_complete_time = time.time()
	                

	elapsed = time.time() - start_time
	if n_images == 1:
		log_file.write("Finished running detector in {}\n".format(humanfriendly.format_timespan(elapsed)))
	else:
		first_image_elapsed = first_image_complete_time - start_time
		remaining_images_elapsed = elapsed - first_image_elapsed
		remaining_images_time_per_image = remaining_images_elapsed/(n_images-1)

		log_file.write("Finished running detector on {} images in {} ({} for the first image, {} for each subsequent image)\n".format(len(images),
			humanfriendly.format_timespan(elapsed),
			humanfriendly.format_timespan(first_image_elapsed),
			humanfriendly.format_timespan(remaining_images_time_per_image)))

	n_boxes = len(boxes)

	n_detections = -1

	for iBox,box in enumerate(boxes):
		n_detections_this_box = box.shape[1]
		assert (n_detections == -1 or n_detections_this_box == n_detections), 'Detection count mismatch'
		n_detections = n_detections_this_box
		assert(box.shape[0] == 1)


	assert(len(scores) == n_images)
	for(iScore,score) in enumerate(scores):
		assert score.shape[0] == 1
		assert score.shape[1] == n_detections
	    
	assert(len(classes) == n_boxes)
	for(iClass,c) in enumerate(classes):
		assert c.shape[0] == 1
		assert c.shape[1] == n_detections
	        
	boxes = np.squeeze(np.array(boxes),axis=1)
	scores = np.squeeze(np.array(scores),axis=1)
	classes = np.squeeze(np.array(classes),axis=1).astype(int)

	assert(len(boxes.shape) == 3)
	assert(boxes.shape[0] == n_images)
	assert(boxes.shape[1] == n_detections)
	assert(boxes.shape[2] == 4)

	assert(len(scores.shape) == 2)
	assert(scores.shape[0] == n_images)
	assert(scores.shape[1] == n_detections)

	assert(len(classes.shape) == 2)
	assert(classes.shape[0] == n_images)
	assert(classes.shape[1] == n_detections)

	return boxes,scores,classes,images

def render_detection_bounding_boxes(detections, image, draw_boxes=True, label_map={},
									classification_label_map={},
									confidence_threshold=0.8, thickness=4,
									classification_confidence_threshold=0.3,
									max_classifications=3):

	display_boxes = []
	display_strs = []
	classes = []

	for detection in detections:

		score = detection['conf']
		if score > confidence_threshold:

			x1, y1, w_box, h_box = detection['bbox']
			display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
			clss = detection['category']
			label = label_map[clss] if clss in label_map else clss
			displayed_label = ['{}: {}%'.format(label, round(100 * score))]

			if 'classifications' in detection:

				log_file.write("Classification start..." + '\n')
				clss = len(bbox_categories) + int(detection['classifications'][0][0])
				classifications = detection['classifications']
				if len(classifications) > max_classifications:
					classifications = classifications[0:max_classifications]
				for classification in classifications:
					p = classification[1]
					if p < classification_confidence_threshold:
						continue
					class_key = classification[0]
					if class_key in classification_label_map:
						class_name = classification_label_map[class_key]
					else:
						class_name = class_key
					displayed_label += ['{}: {:5.1%}'.format(class_name.lower(), classification[1])]

			display_strs.append(displayed_label)
			classes.append(clss)

	display_boxes = np.array(display_boxes)

	if draw_boxes:
		draw_bounding_boxes_on_image(image, display_boxes, classes,
									display_strs=display_strs, thickness=thickness)

	return display_strs


COLORS = [
'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua',  'Azure', 
'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_boxes_on_image(image,
								boxes,
								classes,
								thickness=4,
								display_strs=()):

	boxes_shape = boxes.shape
	if not boxes_shape:
		return
	if len(boxes_shape) != 2 or boxes_shape[1] != 4:
		return
	for i in range(boxes_shape[0]):
		if display_strs:
			display_str_list = display_strs[i]
			draw_bounding_box_on_image(image,
										boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
										classes[i],
										thickness=thickness, display_str_list=display_str_list)


def draw_bounding_box_on_image(image,
								ymin,
								xmin,
								ymax,
								xmax,
								clss=None,
								thickness=4,
								display_str_list=(),
								use_normalized_coordinates=True,
								label_font_size=16):

	if clss is None:
		color = COLORS[1]
	else:
		color = COLORS[int(clss) % len(COLORS)]

	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size
	if use_normalized_coordinates:
		(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
	     							ymin * im_height, ymax * im_height)
	else:
		(left, right, top, bottom) = (xmin, xmax, ymin, ymax)
	draw.line([(left, top), (left, bottom), (right, bottom),
				(right, top), (left, top)], width=thickness, fill=color)

	try:
		font = ImageFont.truetype('arial.ttf', label_font_size)
	except IOError:
		font = ImageFont.load_default()

	display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

	total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

	if top > total_display_str_height:
		text_bottom = top
	else:
		text_bottom = bottom + total_display_str_height

	for display_str in display_str_list[::-1]:

		text_width, text_height = font.getsize(display_str)
		margin = np.ceil(0.05 * text_height)

		draw.rectangle(
			[(left, text_bottom - text_height - 2 * margin), (left + text_width,
		                            							text_bottom)],
			fill=color)

		draw.text(
			(left + margin, text_bottom - text_height - margin),
			display_str,
			fill='black',
			font=font)

		text_bottom -= (text_height + 2 * margin)



def render_bounding_box(box, score, class_label, input_file_name, output_file_name=None,
						confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,linewidth=DEFAULT_LINE_WIDTH):

	output_file_names = []
	if output_file_name is not None:
		output_file_names = [output_file_name]
	scores = [[score]]
	boxes = [[box]]
	categories = render_bounding_boxes(boxes,scores,[class_label],[input_file_name],
										output_file_names,confidence_threshold,linewidth)


def render_bounding_boxes(boxes, scores, classes, input_file_names, output_file_names=[],
						draw_boxes=True, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
						linewidth=DEFAULT_LINE_WIDTH):

	n_images = len(input_file_names)
	iImage = 0
	categories = []

	for iImage in range(0,n_images):

		input_file_name = input_file_names[iImage]

		if iImage >= len(output_file_names):
			output_file_name = ''
		else:
			output_file_name = output_file_names[iImage]

		image = PIL.Image.open(input_file_name).convert("RGB")
		detections = []

		for iBox in range(0,len(boxes)):

			bbox_in = boxes[iImage][iBox]
			bbox = [bbox_in[1],
					bbox_in[0], 
					bbox_in[3]-bbox_in[1],
					bbox_in[2]-bbox_in[0]]

			detections.append({'category':str(classes[iImage][iBox]),
								'conf':scores[iImage][iBox],
								'bbox':bbox})

		class_cat = render_detection_bounding_boxes(detections, image, draw_boxes=draw_boxes,
													confidence_threshold=confidence_threshold, 
													thickness=linewidth,
													label_map=bbox_category_str_id_to_name)
		try:
			if class_cat[0]:
				name_of_dir = class_cat[0][0].partition(':')[0]
				name = os.path.basename(input_file_name)
				output_file_name = os.path.join(os.getcwd(), "detected" , name_of_dir, name)
		except Exception as ex:
			output_file_name = os.path.join(os.getcwd(), "detected/empty", name)
			log_file.write(str(ex))
		
		categories.append(class_cat)
		if len(output_file_name) == 0:
			name, ext = os.path.splitext(input_file_name)
			output_file_name = os.path.join(os.getcwd(), "detected/empty", name)

		image.save(output_file_name)


def load_and_run_detector(model_file, image_file_names, output_dir=None, draw_boxes=True,
						confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
						detection_graph=None):

	if len(image_file_names) == 0:        
		log_file.write('Warning: no files available' + '\n')
		return

	# Load and run detector on target images
	log_file.write('Loading model...' + '\n')
	start_time = time.time()
	if detection_graph is None:
		detection_graph = load_model(model_file)
	elapsed = time.time() - start_time
	log_file.write("Loaded model in {}\n".format(humanfriendly.format_timespan(elapsed)))

	boxes,scores,classes,images = generate_detections(detection_graph,image_file_names)

	assert len(boxes) == len(image_file_names)

	log_file.write('Rendering output...' + '\n')
	start_time = time.time()

	output_full_paths = []
	output_file_names = {}

	if output_dir is not None:

		os.makedirs(output_dir,exist_ok=True)

		for iFn,fullInputPath in enumerate(tqdm(image_file_names)):

			fn = os.path.basename(fullInputPath).lower()            
			name, ext = os.path.splitext(fn)
			fn = "{}{}{}".format(name,DETECTION_FILENAME_INSERT,ext)

			# Since we'll be writing a bunch of files to the same folder, rename
			# as necessary to avoid collisions
			if fn in output_file_names:
				nCollisions = output_file_names[fn]
				fn = str(nCollisions) + '_' + fn
				output_file_names[fn] = nCollisions + 1
			else:
				output_file_names[fn] = 0

		output_full_paths.append(os.path.join(output_dir,fn))
	render_bounding_boxes(boxes=boxes, scores=scores, 
						classes=classes, 
						input_file_names=image_file_names, 
						output_file_names=output_full_paths,
						draw_boxes = draw_boxes,
						confidence_threshold=confidence_threshold)

	elapsed = time.time() - start_time
	log_file.write("Rendered output in {}\n".format(humanfriendly.format_timespan(elapsed)))

	return detection_graph


def main():

	input_dir = r'/home/makar/Documents/netology/cv/data' 
	image_file_names = []
	output_dir = os.path.join(os.getcwd(), "detected/empty")
	for root, dirs, files in os.walk(input_dir):
		for file in files:
			image_file_names.append(os.path.join(root,file))

	log_file.write('Running detector on {} images\n'.format(len(image_file_names)))    
    
	load_and_run_detector(model_file='/home/makar/Documents/netology/cv/md_v4.1.0.pb', 
						image_file_names=image_file_names, 
						confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
						output_dir=output_dir, draw_boxes=False)




if __name__ == '__main__':
    
	main()
log_file.close()



