from distutils.core import setup


setup(name='voxpy',
      version='0.0',
      description='A voxel toolkit.',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      packages=['vox',
                'vox.numpy',
                'vox.torch',
                'vox.numpy.transform_2d',
                'vox.numpy.transform_3d',
                'vox.numpy._transform',
                'vox.torch.transform_2d',
                'vox.torch.transform_3d',
                'vox.torch._transform',
                'vox.viz',
                'vox.utils'],
     )
