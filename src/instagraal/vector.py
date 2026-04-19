#!/usr/bin/env python3

# Vector class
# Author - Ian Johnson | enjalot@gmail.com
# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

import numpy as np


class Vec(np.ndarray):
    props = ["x", "y", "z", "w"]

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if len(obj) < 2 or len(obj) > 4:
            # not a 2,3 or 4 element vector!
            return None
        return obj

    def __array_finalize__(self, obj):
        # this gets called after object creation by numpy.ndarray
        # print "array finalize", obj
        if obj is None:
            return
        # we set x, y etc properties by iterating through ndarray
        for i in range(len(obj)):
            setattr(self, Vec.props[i], obj[i])

    def __array_wrap__(self, out_arr, context=None):
        # this gets called after numpy functions are called on the array
        # out_arr is the output (resulting) array
        for i in range(len(out_arr)):
            setattr(out_arr, Vec.props[i], out_arr[i])
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        desc = """Vec2(data=%(data)s,"""  # x=%(x)s, y=%(y)s)"""
        for i in range(len(self)):
            desc += " %s=%s," % (Vec.props[i], getattr(self, Vec.props[i]))
        desc = desc[:-1]  # cut off last comma
        return desc % {"data": str(self)}

    def __setitem__(self, ind, val):
        # print "SETITEM", self.shape, ind, val
        if self.shape == ():  # dirty hack for dot product
            return
        self.__dict__[Vec.props[ind]] = val
        return np.ndarray.__setitem__(self, ind, val)

    def __setattr__(self, item, val):
        self[Vec.props.index(item)] = val
