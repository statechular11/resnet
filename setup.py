from setuptools import setup

setup(name='resnet',
      version='0.1',
      description='keras-style API to ResNets (ResNet-50, ResNet-101, and ResNet-152)',
      keywords='ResNets ResNet-101 ResNet-152 keras',
      url='https://github.com/statech/resnet',
      download_url='https://github.com/statech/resnet/blob/master/dist/resnet-0.1.tar.gz',
      author='Feiyang Niu',
      author_email='feiyang.niu@gmail.com',
      license='MIT',
      packages=['resnet'],
      install_requires=['keras>=2.0'],
      classifiers=[],
      include_package_data=True,
      zip_safe=False)
