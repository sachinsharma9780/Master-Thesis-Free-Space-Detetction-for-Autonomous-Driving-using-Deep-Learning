import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from skimage import feature
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm




train_imgs = '/home/sachin/Desktop/bdd100k/images/100k/val'
save_edge_maps = '/home/sachin/Desktop/bdd100k/images/100k/val_edge_maps'
print(save_edge_maps)
def create_edge_maps(imgs_path):
	os.chdir(save_edge_maps)
	for image in tqdm(os.listdir(imgs_path)):
		#print('img', os.path.join(imgs_path, image))
		img_obj = Image.open(os.path.join(imgs_path, image))
		grayscale_image = ImageOps.grayscale(img_obj)
		edge_m = feature.canny(np.squeeze(grayscale_image))
		basename, ext = os.path.splitext(image)
		#edge_maps_path = os.path.join(save_edge_maps, img)
		plt.imsave('{}.jpg'.format(basename), edge_m, cmap='gray')
		
		#print('edge_maps', edge_maps.shape)

#for img in tqdm(os.listdir(train_imgs)):
create_edge_maps(train_imgs)

