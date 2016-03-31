#!/usr/bin/env python
from keras_rtst.argparser import get_args
from keras_rtst.main import main


if __name__ == '__main__':
    args = get_args()
    main(args)
