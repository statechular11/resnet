# ResNet

.. image:: https://travis-ci.org/broadinstitute/keras-resnet.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-resnet

Overview
--------
ResNet serves as an extension to [Keras Applications](https://keras.io/applications/) to include 
- ResNet-101
- ResNet-152

The module is based on [Felix Yu](https://github.com/flyyufelix)'s implementation of ResNet-101 and ResNet-152, and his trained weights. Slight modifications have been made to make ResNet-101 and ResNet-152 have consistent API as those pre-trained models in 
[Keras Applications](https://keras.io/applications/). Code is also updated to Keras 2.0.


Installation
------------

.. code-block:: bash

    $ pip install keras-resnet


Usuage
------

.. code-block:: python
    >>> import ResNet
    >>> import numpy as np
    >>> from keras.preprocessing.image import load_img, img_to_array

    >>> #-------------------------------------
    >>> #   Load pre-trained models
    >>> #-------------------------------------
    >>> resnet50  = ResNet.ResNet50(weights='imagenet')
    >>> resnet101 = ResNet.ResNet101(weights='imagenet')
    >>> resnet152 = ResNet.ResNet152(weights='imagenet')

    >>> #-------------------------------------
    >>> #   Helper functions
    >>> #-------------------------------------
    >>> def path_to_tensor(image_path, target_size):
    >>>     image = load_img(image_path, target_size=target_size)
    >>>     tensor = img_to_array(image)
    >>>     tensor = np.expand_dims(tensor, axis=0)
    >>>     return tensor

    >>> #-------------------------------------
    >>> #   Make predictions
    >>> #-------------------------------------
    >>> image_path = 'images/dog.jpeg'
    >>> image_tensor = path_to_tensor(image_path, (224, 224))
    >>> pred_resnet50  = np.argmax(resnet50.predict(image_tensor))
    >>> pred_resnet101 = np.argmax(resnet101.predict(image_tensor))
    >>> pred_resnet152 = np.argmax(resnet152.predict(image_tensor))


![Sample dog image](images/dog.jpeg)

The above dog image is predicted to have
-  257: 'Great Pyrenees' by ResNet-50
-  257: 'Great Pyrenees' by ResNet-101
-  257: 'Great Pyrenees' by ResNet-152


Contact
-------
If you have any questions or encounter any bugs, please contact the author (Feiyang Niu, Feiyang.Niu@gilead.com)


References
----------
- He and etc 2015 Deep Residual Learning for Image Recognition [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Felix Yu's original implementation of ResNet-101 is found [here](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294) and ResNet-152 [here](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6).
