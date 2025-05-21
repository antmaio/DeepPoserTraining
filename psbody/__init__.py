#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.

import os
from os.path import abspath, dirname, expanduser, join

from .mesh import Mesh

texture_path = abspath(join(dirname(__file__), '..', 'data', 'template', 'texture_coordinates'))