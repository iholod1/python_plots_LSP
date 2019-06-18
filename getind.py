#!/usr/bin/env python

import argparse
from read_xdr import get_ind

class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-t', type=float, help='time');
parser.parse_args(namespace=arg)
print(get_ind(arg.t))

