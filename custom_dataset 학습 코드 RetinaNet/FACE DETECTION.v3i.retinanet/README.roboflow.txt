
FACE DETECTION - v3 2024-04-28 3:22pm
==============================

This dataset was exported via roboflow.com on May 2, 2024 at 8:37 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 360 images.
PEOPLE are annotated in retinanet format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -20° to +20° horizontally and -20° to +20° vertically

The following transformations were applied to the bounding boxes of each image:
* Random shear of between -20° to +20° horizontally and -20° to +20° vertically


