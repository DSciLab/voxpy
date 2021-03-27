from distutils.core import setup


setup(name='voxpy',
      version='0.0',
      description='A voxel toolkit.',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      packages=['vox',
                'vox.transform_2d',
                'vox.transform_3d',
                'vox._transform',
                'vox.viz',
                'vox.utils'],
     )
