# ResNet

Overview
--------
ResNet serves as an extension to [Keras Applications](https://keras.io/applications/) to include 
- ResNet-101
- ResNet-152

The module is based on [Felix Yu](https://github.com/flyyufelix)'s implementation of ResNet-101 and ResNet-152, and his trained weights. Slight modifications have been made to make ResNet-101 and ResNet-152 have consistent API as those pre-trained models in 
[Keras Applications](https://keras.io/applications/). Code is also updated to Keras 2.0.

Felix Yu's original implementation of ResNet-101 is found [here](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294).
Felix Yu's original implementation of ResNet-152 is found [here](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6).

Usuage
------
```
import ResNet
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

#-------------------------------------
#   Load pre-trained models
#-------------------------------------
resnet101 = ResNet.ResNet101(weights='imagenet')
resnet152 = ResNet.ResNet152(weights='imagenet')

#-------------------------------------
#   Helper functions
#-------------------------------------
def path_to_tensor(image_path, target_size):
    """
    Read an image from its path, resize it to a specified size (height, width),
    and return a numpy array that is ready to be passed to the `predict` method
    of a trained model

    Parameters
    ----------
    image_path: string
        the path of an image
    target_size: tuple/list
        (height, width) of the image

    Returns
    -------
        a numpy array that is to be readily passed to the `predict` method of
        a trained model
    """
    image = load_img(image_path, target_size=target_size)
    tensor = img_to_array(image)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

#-------------------------------------
#   Make predictions
#-------------------------------------
image_path = 'images/dog.jpeg'
image_tensor = path_to_tensor(image_path)
pred_resnet101 = np.argmax(image_tensor)
pred_resnet152 = np.argmax(image_tensor)

```
