import os

""" Creating a Directory Function"""
def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)