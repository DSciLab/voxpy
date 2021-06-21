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
                'vox.transform',
                'vox.transform.transform_2d',
                'vox.transform.transform_3d',
                'vox.transform.transform_3d.color',
                'vox.transform.transform_3d.geometry',
                'vox.transform._transform',
                'vox.viz',
                'vox.utils'],
     )
