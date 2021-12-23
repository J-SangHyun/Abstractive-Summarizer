# -*- coding: utf-8 -*-


class Module(object):
    def __init__(self, name):
        self.name = name

    def predict(self, indexes, max_length):
        pass

    def predict_batch(self, indexes_list, max_length, batch):
        pass
