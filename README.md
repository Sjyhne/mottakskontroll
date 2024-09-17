# Mottakskontroll

The main idea is to use YoloV8 to count buildings in aerial images. 

## Data

The data must be structured so that we can use the ultralytics package to train the model for object detection of buildings in aerial images.

Todo:
- Find the structure necessary for the ultralytics package to work

## Inference

Due to the nature of the images, being overlapping, we need to do the inference where all tiles are overlapping, and follow these instructions:

- First predict all images, but ensure that the images we are using for inference have, maybe at least 50% overlap with the next image. This might be difficult to fix, but maybe Ben can do it.
- Then after prediction, we should go through all of the bounding boxes and apply a rule that merges bounding boxes if parts of the building has been detected multiple times.

The result of this is that we may be able to get one bounding box per building, and not multiple when we have the case that a single building is present in two or more tiles. 