import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from log.colors import colors, color_id2label
from log.visdom_logger import visdom_logger
# from sbfml.utils.pano import pano

from typing import Union, List, Tuple, Dict

def plot_loss_val(
	logger	: visdom_logger, 
	value	: Union[float, int, torch.Tensor],
	iter	: int,
	name	: str, 
	mode	: str="train", 
	lcolor	: str="blue", 
	markers	: bool=True, 
	marker_symbol	: str="dot", 
	marker_size		: float=5
) -> None:
	'''
	@brief Plot loss values as line plots
	@param	logger			The VisdomLogger instance to use
	@param	loss			The loss value(s) to plot. Must be a dictionary { "value": val, "iter": iteration }
	@param	name			The loss name(s) for each value
	@param	mode			The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	@param	lcolor			The line color for each of the loss values (use Colors dict for ease of use or list of RGB values)
	@param	markers			If True (default) plots markers on line start/emd
	@param	markerSymbol	The marker symbol to use
	@param	markerSize		The size of the marker symbol
	'''
	if isinstance(value, torch.Tensor):
		value = value.detach().cpu().numpy().item()

	lcolor = colors[lcolor]
	lcolor = (np.expand_dims(np.array(lcolor), axis=0) * 255).astype(np.uint8)
	plot_name = f"{name} {mode}"
	
	opts = {
		"title"		: plot_name,
		"xlabel"	: "Iteration",
		"ylabel"	: name,
		"linecolor"	: lcolor,
		"markers"	: markers,
		"markersymbol"	: marker_symbol,
		"markersize"	: marker_size
	}
	logger.instance.line(X=np.array([iter]), Y=np.array([value]), env=logger.env_name, win=plot_name, opts=opts, update="append")


def plot_text(
	logger	: visdom_logger, 
	text	: str, 
	title	: str, 
	mode	: str="Train"
) -> None:
	'''
	@brief Plot Text
	@param	logger			The VisdomLogger instance to use
	@param	text	The text to plot (can be html)
	@param 	title	The window title
	'''
	plot_name = f"{title} {mode}"
	logger.instance.text(text, env=logger.env_name, win=plot_name)


def plot_img(
	logger	: visdom_logger, 
	img		: torch.Tensor, 
	title	: str, 
	mode	: str="Train"
) -> None:
	'''
	@brief Plot an Image
	@param	logger			The VisdomLogger instance to use
	@param	img		The image to plot (3-channel) (minibatch of torch.Tensor - if the minibatch is larger than 1, then only the first image in the batch will be displayed)
	@param	title	The title of the window to plot the image
	@param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	'''
	b, c, h, w = img.size()
	if c == 1:
		img = torch.repeat_interleave(img, 3, dim=1)
	elif c != 3:
		raise RuntimeError("PlotImage() can display only 3-channel images.")
	
	disp = img.detach().cpu().numpy()
	disp = disp[0, :, :, :]
	plot_name = f"{title} {mode}"
	opts = {
		"title": plot_name
	}
	logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)


def plot_img_bbox(
	logger	: visdom_logger, 
	img		: torch.Tensor, 
	bbox	: torch.Tensor, 
	title	: str, 
	mode	: str="Train"
) -> None:
	'''
	@brief Plot an Image with a bounding box overlayed
	@param	logger	The VisdomLogger instance to use
	@param	img		The image to plot (3-channel) (minibatch of torch.Tensor - if the minibatch is larger than 1, then only the first image in the batch will be displayed)
	@param	bbox	The bounding box to plot (can be multiple bounding boxes)
	@param	title	The title of the window to plot the image
	@param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	'''
	b, c, h, w = img.size()
	# if bbox is in format N x 4 (xmin, ymin, xmax, ymax)
	assert c == 3, "plot_img_bbox() can display only 3-channel images."
	disp = img[0, :, :, :]
	disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)

	if bbox.ndim == 2:
		bb, bw = bbox.size()
		assert bw == 4, "plot_img_bbox() accepts bbox in format [xmin, ymin, xmax, ymax]"

		cbbox = bbox[0, :].detach().cpu().numpy()
		disp = cv2.rectangle(disp, (cbbox[0], cbbox[1]), (cbbox[2], cbbox[3]), colors["lime"], 2)
	elif bbox.ndim == 3:
		bb, bc, bw = bbox.size()
		assert bw == 4, "plot_img_bbox() accepts bbox in format [xmin, ymin, xmax, ymax]"
		
		for bboxidx, color in zip(range(bc), colors):
			cbbox = bbox[0, bboxidx, :].detach().cpu().numpy()
			disp = cv2.rectangle(disp, (cbbox[0], cbbox[1]), (cbbox[2], cbbox[3]), colors[color], 2)

	# disp = img[0, :, :, :]
	# disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)
	disp = disp.transpose(2, 0, 1)
	plot_name = f"{title} {mode}"
	opts = {
		"title": plot_name
	}
	logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)


def plot_detection(
	logger			: visdom_logger,
	img				: torch.Tensor,
	bbox			: torch.Tensor,
	classes			: torch.Tensor,
	class_mapping	: Dict,
	title			: str,
	mode			: str
) -> None:
	'''
	@brief Plot an Image with a bounding box overlayed
	@param	logger	The VisdomLogger instance to use
	@param	img		The image to plot (3-channel) (minibatch of torch.Tensor - if the minibatch is larger than 1, then only the first image in the batch will be displayed)
	@param	bbox	The bounding box to plot (can be multiple bounding boxes)
	@param	title	The title of the window to plot the image
	@param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	'''
	b, c, h, w = img.size()
	# if bbox is in format N x 4 (xmin, ymin, xmax, ymax)
	assert c == 3, "plot_img_bbox() can display only 3-channel images."
	disp = img[0, :, :, :]
	disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)

	if bbox.ndim == 2:
		bb, bw = bbox.size()
		assert bw == 4, "plot_img_bbox() accepts bbox in format [xmin, ymin, xmax, ymax]"

		# cbbox = bbox[0, :].detach().cpu().numpy()
		# cbbox = bbox.detach().cpu().numpy()
		for bboxidx, color in zip(range(bbox.size(0)), colors.keys()):
			cbbox = bbox[bboxidx, :].detach().cpu().numpy()
			color = (np.asarray(list(colors[color])) * 255).astype(np.uint8)
			color = tuple(color.tolist())
			disp = cv2.rectangle(disp, (cbbox[0], cbbox[1]), (cbbox[2], cbbox[3]), color, 2)
			classname = classes[bboxidx]
			classname = class_mapping[classname.item()]
			disp = cv2.putText(disp, classname, (cbbox[0], cbbox[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.3, color=(255, 255, 255))
		# disp = cv2.putText(disp, "Yoooo", (cbbox[0], cbbox[1]))
	elif bbox.ndim == 3:
		bb, bc, bw = bbox.size()
		assert bw == 4, "plot_img_bbox() accepts bbox in format [xmin, ymin, xmax, ymax]"
		
		for bboxidx, color in zip(range(bc), colors.keys()):
			cbbox = bbox[0, bboxidx, :].detach().cpu().numpy()
			color = (np.asarray(list(colors[color])) * 255).astype(np.uint8)
			color = tuple(color.tolist())
			disp = cv2.rectangle(disp, (cbbox[0], cbbox[1]), (cbbox[2], cbbox[3]), color, 2)
			classname = classes[:, bboxidx]
			classname = class_mapping[classname.item()]
			disp = cv2.putText(disp, classname, (cbbox[0], cbbox[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.3, color=(255, 255, 255))

	disp = disp.transpose(2, 0, 1)
	plot_name = f"{title} {mode}"
	opts = {
		"title": plot_name
	}
	if logger is not None:
		logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)
	disp = disp.transpose(1, 2, 0)
	disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
	return disp
	


def plot_img_quad(
	logger	: visdom_logger, 
	img		: torch.Tensor, 
	quad	: torch.Tensor, 
	title	: str, 
	mode	: str="Train"
) -> None:
	##
	# @brief Plot an Image with a bounding box overlayed
	# @param	logger	The VisdomLogger instance to use
	# @param	img		The image to plot (3-channel) (minibatch of torch.Tensor - if the minibatch is larger than 1, then only the first image in the batch will be displayed)
	# @param	bbox	The bounding box to plot (can be multiple bounding boxes)
	# @param	title	The title of the window to plot the image
	# @param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	b, c, h, w = img.size()
	# if bbox is in format N x 4 (xmin, ymin, xmax, ymax)
	assert c == 3, "plot_img_quad() can display only 3-channel images."
	assert quad.ndim == 3, "plot_img_quad() accepts quads of dimensions N x 2 x 4"
	
	qb, qc, qw = quad.size()
	
	disp = img[0, :, :, :]
	disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)
	
	q = quad[0, :, :].detach().cpu().numpy()
	q = q.reshape(qc, qw).astype(np.int)
	
	disp = cv2.drawContours(disp, [q], contourIdx=-1, color=colors["blue"], thickness=3)
	disp = disp.transpose(2, 0, 1)
	
	plot_name = f"{title} {mode}"
	opts = { "title": plot_name }
	logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)


def plot_img_points(
	logger	: visdom_logger, 
	img		: torch.Tensor, 
	points	: torch.Tensor, 
	title	: str, 
	mode	: str="Train"
) -> None:
	# TODO: Use list of point tensor for multiple point-set ploting
	##
	# @brief Plot an Image with a bounding box overlayed
	# @param	logger	The VisdomLogger instance to use
	# @param	img		The image to plot (3-channel) (minibatch of torch.Tensor - if the minibatch is larger than 1, then only the first image in the batch will be displayed)
	# @param	bbox	The bounding box to plot (can be multiple bounding boxes)
	# @param	title	The title of the window to plot the image
	# @param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	b, c, h, w = img.size()
	# if bbox is in format N x 4 (xmin, ymin, xmax, ymax)
	assert c == 3, "plot_img_points() can display only 3-channel images."
	disp = img[0, :, :, :]
	disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)

	if isinstance(points, torch.Tensor) or (isinstance(points, List) and len(points) > 1):
		b, c, w = points.size()
		points = points[0, :, :].detach().cpu().numpy()
		color = np.array(colors["lime"]) * 255
		for pointIdx in range(c):
			point = points[pointIdx, :]
			disp = cv2.circle(disp, tuple(point), radius=2, color=color, thickness=-1)

		disp = disp.transpose(2, 0, 1)
		plot_name = f"{title} {mode}"
		opts = { "title": plot_name }
		logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)


def plot_compare_img_points(
	logger: visdom_logger, 
	img: torch.Tensor, 
	points: torch.Tensor, 
	compare_points, 
	title, 
	mode="train"
) -> None:
	b, c, h, w = img.size()
	# if bbox is in format N x 4 (xmin, ymin, xmax, ymax)
	assert c == 3, "plot_compare_img_points() can display only 3-channel images."
	disp = img[0, :, :, :]
	disp = np.ascontiguousarray(disp.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)

	b, c, w = points.size()
	points = points[0, :, :].detach().cpu().numpy()
	comp_points = compare_points[0, :, :].detach().cpu().numpy()
	for point_idx in range(c):
		point = points[point_idx, :]
		comp_point = comp_points[point_idx, :]
		disp = cv2.circle(disp, tuple(point), radius=3, color=colors["lime"], thickness=-1)
		disp = cv2.circle(disp, tuple(comp_point), radius=3, color=colors["red"], thickness=-1)

	disp = disp.transpose(2, 0, 1)
	plot_name = f"{title} {mode}"
	opts = {
		"title": plot_name
	}
	logger.instance.image(disp, env=logger.env_name, opts=opts, win=plot_name)

##
# @brief Plot images on a grid
# @param	logger			The VisdomLogger instance to use
# @param	imgs	The images to plot (3-channel) (minibatch of torch.Tensor - if minibatch is larger than 1, then batchCount images will be displayed)
# @params	title	The title of the window to plot the images
# @param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
# @param	nrow	The number of images in a row
# @param	pad		The padding between the images
def plot_images(logger, imgs, title, mode="Train", nrow=4, pad=2):
	raise NotImplementedError()


def plot_map(
	logger	: visdom_logger, 
	map		: torch.Tensor, 
	title	: str, 
	mode	: str="Train", 
	cmap	: str="Viridis"
) -> None:
	##
	# @brief Plot a single channel image using a colormap
	# @param	logger	The VisdomLogger instance to use
	# @param	map		The map to plot (1-channel) (minibatch of torch.Tensor - if the minibatch size is larger than 1, then the first map in the batch will be displayed)
	# @param	title	The title of the window to plot the map
	# @param	mode	The experiment mode ("Train", "Evaluation", "Test" or whatever - it used for window naming on Visdom)
	# @param	cmap	The colormap to use
	b, c, h, w = map.size()
	if c != 1:
		raise Warning("PlotMap() can display only single channel images.")
	disp = None
	disp = map[0, 0, :, :]
	disp = torch.flip(disp.detach().cpu(), [0]).numpy()
	plot_name = f"{title} {mode}"
	opts = {
		"title": plot_name,
		"colormap": cmap
	}
	logger.instance.heatmap(disp, win=plot_name, env=logger.env_name, opts=opts)


def plot_activations(
	logger	: visdom_logger, 
	val		: torch.Tensor, 
	title	: str, 
	mode	: str="Train", 
	cmap	: str="Electric"
) -> None:
	##
	# @brief Plot a tensor that represents a module's activations.
	# The input tensor can be of arbitrary size (batch and channel-wise).
	# This function plots the channel-wise mean value of the input tensor
	# @param	logger		The VisdomLogger instance to use
	# @param	activations	The acitvations tensor to plot
	# @param	title		The plot's title
	# @param	mode		The experiment mode ("Train", "Validation", "Test" or whatever you may like)
	# @param	cmap		The colormap to use
	b, c, h, w = val.size()
	plotName = f"{title} {mode}"
	opts = {
		"title": plotName,
		"colormap": cmap
	}
	# reverse the order of rows (height) because in visdom's heatmap (0, 0) is the bottom left point of the map
	activations = torch.flip(val, dims=[2])
	activations = activations[0, :, :, :]
	num_elmnts = c * h * w
	activations = torch.sum(activations, dim=0)
	activations = (activations - activations.min()) / (activations.max() - activations.min())
	logger.instance.heatmap(activations, win=plotName, env=logger.env_name, opts=opts)


def plot_aggregated_activations(
	logger		: visdom_logger, 
	activations	: torch.Tensor, 
	size		: Union[List[int], Tuple[int, int]], 
	title		: str, 
	mode		: str="Train", 
	cmap		: str="Electric"
) -> None:
	##
	# @brief Plot an aggregation of activations
	# The input is a list/tuple of activation tensors
	# @parram	logger		The VisdomLogger instance to use
	# @param	activations	The aggregation of activation maps to plot
	# @param	title		The window title
	# @param	mode		The experiment mode ("Train", "Test", "Validation" or whatever)
	# @param	cmap		The color map to use
	plotName = f"{title} {mode}"
	opts = {
		"title": plotName,
		"colormap": cmap
	}
	upsample = nn.UpsamplingBilinear2d(size=size)
	aggregated = torch.zeros(size, dtype=torch.float32)
	for activation in activations:
		activation = activation
		b, c, h, w = activation.size()
		activation = torch.flip(activation, dims=[2])
		activation = upsample(activation)
		activation = activation[0, :, :, :]
		activation = torch.sum(activation, dim=0)
		aggregated += activation.cpu()
	aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
	logger.instance.heatmap(aggregated, win=plotName, env=logger.env_name, opts=opts)


def plot_aggregated_activations_over(
	logger		: visdom_logger, 
	img			: torch.Tensor, 
	activations	: torch.Tensor, 
	size		: Union[List[int], Tuple[int, int]], 
	title		: str, 
	mode		: str="Train", 
	cmap		: str="Electric"
) -> None:
	##
	# @brief Plot an aggregation of activations
	# The input is a list/tuple of activation tensors
	# @parram	logger		The VisdomLogger instance to use
	# @param	activations	The aggregation of activation maps to plot
	# @param	title		The window title
	# @param	mode		The experiment mode ("Train", "Test", "Validation" or whatever)
	# @param	cmap		The color map to use
	plotName = f"{title} {mode}"
	opts = {
		"title": plotName,
		"colormap": cmap
	}
	aggregated = torch.zeros(size, dtype=torch.float32)
	for activation in activations:
		b, c, h, w = activation.size()
		activation = torch.flip(activation, dims=[2])
		activation = activation[0, :, :, :]
		activation = torch.sum(activation, dim=0)
		activation = F.interpolate(activation, size=size, mode="linear")
		aggregated += activation
	aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
	logger.instance.heatmap(aggregated, win=plotName, env=logger.env_name, opts=opts)


def plot_img_overlay(
	logger	: visdom_logger, 
	image	: torch.Tensor, 
	overlay	: torch.Tensor, 
	weight	: float=0.3, 
	title	: str=None, 
	mode	: str="Train"
) -> None:
	##
	# @brief plot an image with an overlay.
	# @param	logger	`visdom_logger`. A `visdom_logger` instance 
	# @param	image	`torch.tensor`. The image to plot.
	# @param	overlay	`torch.tensor`. The overlay to plot.
	# @param	weight	`float`. The transparency weight of the overlay (in [0.0, 1.0]).
	# @param	title	`str`. The title of the plot.
	# @param	mode	`str`. The experiment mode
	plotName = f"{title} {mode}"
	isOneHot = False
	if overlay.ndim == 3:
		overlay = overlay.unsqueeze(1)
		isOneHot = True
	cb, cc, ch, cw = image.size()
	ob, oc, oh, ow = overlay.size()

	for idx in range(logger.batch_count):
		img = image[idx, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
		over = None
		if not isOneHot:
			over = torch.argmax(overlay, 1)[idx, :, :].unsqueeze(2).cpu().numpy().astype('uint8')
		else:
			over = overlay[idx, 0, :, :].unsqueeze(2).cpu().numpy().astype('uint8')
			oc = np.unique(over).max()
		over = np.repeat(over, 3, axis=2)
		for label in range(oc):
			color = np.array(colors[color_id2label[label+15]])
			idx = over == (label, label, label)
			idx = idx[:, :, 0] & idx[:, :, 1] & idx[:, :, 2]
			over[idx, :] = color
			# over[np.where((over==[label, label, label]).all(axis=2))] = color
		over = over.astype('float32')
		over = (over - over.min()) / (over.max() - over.min())
		res = cv2.addWeighted(over, weight, img, 1.0-weight, 0).transpose(2, 0, 1)
		plotName += f" {idx}"
		opts = {"title": plotName}
		logger.instance.image(res, win=plotName, env=logger.env_name, opts=opts)


def plot_scatter(
	logger	: visdom_logger, 
	val		: torch.Tensor, 
	title	: str, 
	mode	: str="Train"
) -> None:
	##
	# @brief Plot a scatter diagram
	# @param	logger	The VisdomLogger instance to use
	# @param	val		The values to plot (must be 2d)
	# @param	title	The plot title
	# @param	mode	The experiment mode ("Train", "Validation", "Test" or whatever you want)
	plotName = f"{title} {mode}"
	opts = {
		"title": f"{title} {mode}"
	}
	logger.instance.scatter(val.unsqueeze(0), win=plotName, env=logger.env_name, opts=opts)


# def plot_pano_lines_from_points(
# 	logger	: visdom_logger, 
# 	img		: torch.Tensor, 
# 	points	: torch.Tensor, 
# 	title	: str, 
# 	mode	: str="train"
# ) -> None:
# 	##
# 	# @brief
# 	b, c, h, w = img.size()
# 	plot_name = f"{title} {mode}"
# 	pano_utils = pano(w, h)
# 	disp = img[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
# 	point_disp = points[0, :, :].detach().cpu().numpy()
# 	point_all = [point_disp]
# 	for i in range(len(point_disp)):
# 		point_all.append(point_disp[i, :])
# 		point_all.append(point_disp[(i + 2) % len(point_disp), :])
# 	point_all = np.vstack(point_all)
# 	rs, cs = pano_utils.line_idx_from_coords(point_all)
# 	rs = np.array(rs)
# 	cs = np.array(cs)

# 	pano_edge_c = disp.astype(np.uint8)
# 	for dx, dy in [[-1, 0], [1, 0], [0, 0], [0, 1], [0, -1]]:
# 		pano_edge_c[np.clip(rs + dx, 0, h - 1), np.clip(cs + dy, 0, w - 1), 0] = 0
# 		pano_edge_c[np.clip(rs + dx, 0, h - 1), np.clip(cs + dy, 0, w - 1), 1] = 0
# 		pano_edge_c[np.clip(rs + dx, 0, h - 1), np.clip(cs + dy, 0, w - 1), 2] = 255
# 	opts = { "title": title }
# 	pano_edge_c = (pano_edge_c - pano_edge_c.min()) / (pano_edge_c.max() - pano_edge_c.min())
# 	pano_edge_c = pano_edge_c.transpose(2, 0, 1)
# 	logger.instance.image(pano_edge_c, win=plot_name, env=logger.env_name, opts=opts)


def plot_semantic_map(
	logger	: visdom_logger, 
	map		: torch.Tensor, 
	title	: str, 
	mode	: str="train"
) -> None:
	##
	# @brief
	if isinstance(map, list):
		if len(map[0].size()) == 3:
			map = map[0].unsqueeze(0) 
		else:
			map = map[0].permute(1, 0, 2, 3)
	# else:
	# 	map = map.permute(1, 0, 2, 3)
	b, c, h, w = map.size()
	
		# raise  RuntimeError(f"The input segmenation map should have 1 channel.")
	plot_name = f"{title} {mode}"
	class_ids = map.detach().cpu().numpy()[0, :, :, :]
	
	if class_ids.shape[0] != 1:
		class_ids = np.argmax(class_ids, axis=0)
		class_ids = np.expand_dims(class_ids, 0)

	classes_un = np.unique(class_ids)
	disp = np.zeros([h, w, 3])
	for class_id in classes_un:
		# + 15 to get a "better" set of colors from our palette starting from 'black
		color = np.array(colors[color_id2label[class_id + 15]])
		idx = class_id == class_ids
		idx = np.squeeze(idx, axis=0)
		disp[idx, :] = color
			
	# disp = (disp - disp.min()) / (disp.max() - disp.min())
	disp = disp.transpose(2, 0, 1) * 255
	disp = disp.astype(np.uint8)
	opts = { "title": title }

	if logger is not None:
		logger.instance.image(disp, win=plot_name, env=logger.env_name, opts=opts)
	disp = disp.transpose(1, 2, 0)
	disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
	return disp

def plot_normals_map(
	logger	: visdom_logger, 
	normals	: torch.Tensor, 
	title	: str, 
	mode	: str="train", 
	scale_fn=None
) -> None:

	b, c, h, w = normals.size()
	plot_name = f"{title} {mode}"
	disp = normals.detach().cpu().numpy()[0, :, :, :]
	disp = 128 * (disp + 1)
	disp = disp.astype(np.uint8)
	disp = (disp - disp.min()) / (disp.max() - disp.min())
	opts = { "title": title }
	logger.instance.image(disp, win=plot_name, env=logger.env_name, opts=opts)

def plot_shape_map(
	logger: visdom_logger,
	shape_map: torch.Tensor,
	title: str,
	mode: str='train'
) -> None:
	'''
	'''
	b, c, h, w = shape_map.size()
	plot_name = f"{title} {mode}"
	disp = shape_map.detach().cpu().numpy()[0, :, :, :]
	# shapes are in [-1, 1] so we trasform them in order to lie in [0, 1]
	disp = (disp - disp.min()) / (disp.max() - disp.min())
	opts = { 'title': title }
	logger.instance.image(disp, win=plot_name, env=logger.env_name, opts=opts)

def plot_normals_map_no_scale(
	logger: visdom_logger,
	normals_map: torch.Tensor,
	title: str,
	mode: str='train'
) -> None:
	plot_shape_map(logger, normals_map, title, mode)