#!/usr/bin/python

if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
