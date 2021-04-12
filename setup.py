from distutils.core import setup
import vox


setup(name='voxpy_dist',
      version=vox.__version__,
      description='A voxel toolkit.',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'numpy',
            'torch'
      ],
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
