import random
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d_%H%M%S')
    code = ''.join(random.choices('0123456789qwertyui' + \
                                'opasdfghjklzxcvbnmQWERT' + \
                                'YUIOPASDFGHJKLZXCVBNM', k=5))
    return f'{code}-{date_str}'


__version__ = f'0.1-{gen_code()}'
