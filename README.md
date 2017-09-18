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

![Sample dog image](images/dog.jpeg =250x)

The above dog image is predicted to have
-  257: 'Great Pyrenees' by ResNet-101
-  257: 'Great Pyrenees' by ResNet-152
