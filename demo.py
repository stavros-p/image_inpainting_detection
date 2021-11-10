import argparse
from re import L
import cv2
import torch
from torchvision import transforms
import os
import torch.nn as nn
from train import InpaintingForensics
import numpy as np

g_norm_trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def open_image(img_path: str) -> torch.Tensor:
	'''
	'''
	assert os.path.exists(img_path), f"The specified path '{img_path}' does not exist."
	img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
	tensor = (torch.from_numpy(img_np).transpose(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255)
	return g_norm_trans(tensor)
	

def load_model(network: str, weights: str) -> nn.Module:
	'''
	'''
	assert network in ['hp', 'hrnet'], f"The specified method is not correct. Expected one of ['hp', 'hrnet'], got {network}."
	assert os.path.exists(weights), f"The specified weights path doesn't exist."
	net = InpaintingForensics.create(network, weights)
	net.eval()
	return net

def main(input_path: str, method: str, weights: str, output_dir: str, out_dir: str, use_gpu: bool=False) -> None:
	'''
	'''
	assert os.path.exists(output_dir), f"The specified output path '{output_dir}' doesn't exist."

	img = open_image(input_path)
	network = load_model(method, weights)

	if use_gpu:
		network = network.to(device='cuda:0')
	
	result = network(img)
	result = result.squeeze(0).transpose(1, 2, 0).cpu().numpy()
	result = (result * 255).astype(np.uint8)
	result = cv2.applyColorMap(result, cv2.COLORMAP_VIRIDIS)
	cv2.imwrite(os.path.join(output_dir, 'result.png'), result)
	print('done.')


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('in_path', type=str, help="The path to the input image file.")
	parser.add_argument('w_path', type=str, help="The filepath to the weight data.")
	parser.add_argument('method', type=str, help="The architecture type to use")
	parser.add_argument('--gpu', action="store_true", help="Set this flag to use the gpu", default=False)
	parser.add_argument('output', type=str, help="The path to the output directory.")

	args = parser.parse_args()
	main(args.in_path, args.method, args.w_path, args.output, args.gpu)