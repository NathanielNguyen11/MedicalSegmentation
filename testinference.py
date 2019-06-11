#@title Imports
import sys
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import test
import tensorflow as tf
import cv2
import argparse

# IMAGE = '/home/ubuntu/Desktop/data/skin/test/image1/image/ISIC_0015462.png'  #@param {type:"string"}
IMDIR = '/home/ubuntu/Desktop/CVC-612_1/1'
RE_DIR = '/home/ubuntu/Desktop/CVC-612_1/1'
OTDIR = '/home/ubuntu/Desktop/CVC-612_1/2_pre'
MODEL_PATH= '/media/ubuntu/Quang/best_result/logs1.tar.gz'


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--im_dir',type = str,default=IMDIR, help='Directory with original images.')
  parser.add_argument('--fol_refer',type = str,default=RE_DIR, help='Directory with original images.')
  parser.add_argument('--model_dir',type = str,default=MODEL_PATH, help='Directory with original images.')
  parser.add_argument('--output_dir',type = str,default=OTDIR, help='Directory with original images.')
  # parser.add_argument('--time',type=int, help='How many time do you want to resize')
  return parser.parse_args(argv)
def resize_image(image, width, height):
  im_processed = cv2.resize(image,(height,width))
  return im_processed
def run_visualization(url_im,output_dir,name,refim_path):
  """Inferences DeepLab model and visualizes result."""
  refim = cv2.imread(refim_path)
  # Height = refim.shape[0]
  # Width  = refim.shape[1]

  MODEL = test.DeepLabModel(MODEL_PATH)
  try:
    # f = urllib.request.urlopen(url)
    # jpeg_str = url.read()
    original_im = Image.open(url_im)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url_im)
    return

  print('running deeplab on image %s...' % url_im)
  resized_im, seg_map = MODEL.run(original_im)
  seg_image,image = test.vis_segmentation(resized_im, seg_map)
  # seg_im,image = test.vis_segmentation(resized_im, seg_map)
  # cv2.imshow('aaaa',seg_im)
  # seg_image_final = resize_image (seg_image, Height, Width)
  name = name + '.png'
  processed_path = os.path.join(output_dir,name)
  cv2.imwrite('%s'%processed_path,seg_image)


def main(args):
  image_dir = args.im_dir
  output_dir = args.output_dir
  refdir = args.fol_refer

  check = os.path.isdir(args.output_dir)

  if check !=True:
    os.makedirs(output_dir,0777)
  else:
    pass
    
  images = os.listdir(image_dir)

  for image in images:
    name = image[:-4]
    name1 = image[:-4]+'.tif'
    print image
    image_path = os.path.join(image_dir,image)
    refim_path = os.path.join(refdir,name1)
    print refim_path
    run_visualization(image_path,output_dir,name,refim_path)


if __name__ == '__main__':
  # for 
  main(parse_arguments(sys.argv[1:]))
  # run_visualization(IMAGE)