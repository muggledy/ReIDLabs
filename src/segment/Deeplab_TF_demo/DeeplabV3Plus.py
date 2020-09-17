'''
This demo provided by zhiyuan luo
'''

import os
import tarfile
import numpy
import tensorflow
from six.moves import urllib
from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot
import cv2
from IPython import embed

import scipy.io as scio

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tensorflow.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tensorflow.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tensorflow.import_graph_def(graph_def, name='')

    self.sess = tensorflow.compat.v1.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [numpy.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = numpy.zeros((256, 3), dtype=int)
  ind = numpy.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if numpy.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image,seg_map,labelNames):
  """Visualizes input image, segmentation map and overlay view."""
  pyplot.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  pyplot.subplot(grid_spec[0])
  pyplot.imshow(image)
  pyplot.axis('off')
  pyplot.title('input image')

  pyplot.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(numpy.uint8)
  pyplot.imshow(seg_image)
  pyplot.axis('off')
  pyplot.title('segmentation map')

  pyplot.subplot(grid_spec[2])
  pyplot.imshow(image)
  pyplot.imshow(seg_image, alpha=0.7)
  pyplot.axis('off')
  pyplot.title('segmentation overlay')

  unique_labels = numpy.unique(seg_map)
  ax = pyplot.subplot(grid_spec[3])
  pyplot.imshow(
      FULL_COLOR_MAP[unique_labels].astype(numpy.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  pyplot.yticks(range(len(unique_labels)), labelNames[unique_labels])
  pyplot.xticks([], [])
  ax.tick_params(width=0.0)
  pyplot.grid('off')
  pyplot.show()

def run_visualization(deepLabModel,imageFileName,labelNames):
  inputImage = Image.open(imageFileName)
  print('running deeplab on image %s...' % imageFileName)
  resized_im, seg_map = deepLabModel.run(inputImage)
  vis_segmentation(resized_im, seg_map,labelNames)

def generateOverlayComparisonImage(rawImage, imageMask, comparisonImageFileName):
    imageOverlay = getImageOverlay(rawImage, imageMask)
    concatenatedImage = numpy.concatenate((rawImage, imageMask, imageOverlay), axis=1)
    cv2.imwrite(comparisonImageFileName, cv2.cvtColor(concatenatedImage, cv2.COLOR_BGR2RGB))

def generateForegroundAndBackGroundComparisonImage(rawImage, foregroundImage,backgroundImage, comparisonImageFileName):
    concatenatedImage = numpy.concatenate((rawImage, foregroundImage, backgroundImage), axis=1)
    cv2.imwrite(comparisonImageFileName, cv2.cvtColor(concatenatedImage, cv2.COLOR_BGR2RGB))

def getImageOverlay(rawImage,imageMask):
    imageOverlay = cv2.addWeighted(rawImage, 0.5, imageMask, 0.5, 0)
    return imageOverlay

def getRawImageAndMask(inputImage):
    resized_im, seg_map = deepLabModel.run(inputImage)
    rawImage=numpy.array(resized_im)
    return rawImage, seg_map

def getForegroundFilter(seg_map):
    foregroundFilter=numpy.where(seg_map != 0, 1, 0)
    return foregroundFilter

def getForegroundImage(rawImage, seg_map):
    foregroundFilter=getForegroundFilter(seg_map)
    foreGroundImage=numpy.empty((seg_map.shape[0],seg_map.shape[1],3),dtype='uint8')
    foreGroundImage[:,:,0]=rawImage[:,:,0]*foregroundFilter
    foreGroundImage[:, :, 1] = rawImage[:, :, 1] * foregroundFilter
    foreGroundImage[:, :, 2] = rawImage[:, :, 2] * foregroundFilter
    return foreGroundImage

def getBackgroundImage(rawImage, seg_map):
    backgroundFilter=numpy.where(seg_map == 0, 1, 0)
    backGroundImage=numpy.empty((seg_map.shape[0],seg_map.shape[1],3),dtype='uint8')
    backGroundImage[:,:,0]=rawImage[:,:,0]*backgroundFilter
    backGroundImage[:, :, 1] = rawImage[:, :, 1] * backgroundFilter
    backGroundImage[:, :, 2] = rawImage[:, :, 2] * backgroundFilter
    return backGroundImage


if __name__=='__main__':
    labelNames = numpy.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])
    FULL_LABEL_MAP = numpy.arange(len(labelNames)).reshape(len(labelNames), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    modelName = 'mobilenetv2_coco_voctrainaug'
    modelFileName = 'deeplab_model.tar.gz'
    modelDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Data')
    modelFileNameWithFullPath = os.path.join(modelDirectory, modelFileName)
    if not os.path.exists(modelFileNameWithFullPath):
        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[modelName],modelFileNameWithFullPath)
        print('download completed!')

    print('loading DeepLab model...')
    deepLabModel = DeepLabModel(modelFileNameWithFullPath)
    print('model loaded successfully!')


    outputDirectory=os.path.join(os.path.dirname(os.path.realpath(__file__)),'Output')
    inputDirectory=os.path.join(os.path.dirname(os.path.realpath(__file__)),'Input')

    inputImageFileNames=os.listdir(inputDirectory)
    for imageFileName in inputImageFileNames:

        imageNameWithFullPath=os.path.join(inputDirectory,imageFileName)
        imageFileNameWithoutSuffix = os.path.basename(imageFileName).split('.')[0]

        inputImage = Image.open(imageNameWithFullPath)
        print('running deeplab on image %s...' % imageFileName)
        rawImage, seg_map = getRawImageAndMask(inputImage)

        #embed()
        #imageColorMask = label_to_color_image(seg_map)

        #rawImageFileName = outputDirectory + '/' + imageFileNameWithoutSuffix + '_raw.png'
        #cv2.imwrite(rawImageFileName, cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB))
        foreGroundFilterFileName=outputDirectory + '/' + imageFileNameWithoutSuffix + '_filter.mat'
        foregroundFilter=getForegroundFilter(seg_map)
        scio.savemat(foreGroundFilterFileName,{'foregroundFilter':foregroundFilter})
        
        foregroundImage = getForegroundImage(rawImage, seg_map)
        foregroundImageFileName = outputDirectory + '/' + imageFileNameWithoutSuffix + '_foreground.png'
        cv2.imwrite(foregroundImageFileName, cv2.cvtColor(foregroundImage, cv2.COLOR_BGR2RGB))
