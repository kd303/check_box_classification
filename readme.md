## Checkbox Classification

A simple Binary classification of checkboxes using simple CNN network
Following are the objectivces:

1. Understand CNN, how to calculate dimensions [see ```CheckBoxClassification```](./custom_image_binary_classification.py) - no dropout/batchnorm are used - it not to learn tunning a model. 
    a. Calulating Conv2d ouputs
    b. Cacluating MaxPool2d ouputs
    c. usage of model.train() and model.eval() -> to ensure if batchnorm, drop outs behaved differently for training and evaluation
2. BCEWithLogLoss function usage, instead of [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), it has sigmoid as well as log function init, and can be quite confusing using it. 

3. usage of [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html)

## Data

1. Training Data for [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html)
    - 0 labled images -> check_box_classification/train/checked
    - 1 labled images -> check_box_classification/train/unchecked
2. Validation data
    - 0 labled images -> check_box_classification/test/checked
    - 1 labled images -> check_box_classification/test/unchecked

## Code

1. To run from ```conda```:

Change the training and validation folder path as per convinience

```conda run -n base --no-capture-output --live-stream python test_checked_unchecked.py --train_data_path check_box_classification/train --validation_data_path check_box_classification/test --num_epoch 2 --lr 0.001 --batch_size 8 ```

from ```python```

```python test_checked_unchecked.py --train_data_path /check_box_classification/train --validation_data_path D:/code/py/check_box_classification/test --num_epoch 2 --lr 0.001 --batch_size 8 ```
