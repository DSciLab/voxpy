from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    return f'dev{date_str}'


__version__ = f'0.0.1-{gen_code()}'


setup(name='voxpy_dist',
      version=__version__,
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
