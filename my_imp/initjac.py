import sys
import yaml
import os.path as osp

def insert_to_path(pth):
    if pth not in sys.path:
        sys.path.insert(0, pth)

insert_to_path('Jacinle')

vendors = yaml.load(open('Jacinle/jacinle.yml', 'r').read())
for v in vendors['vendors'].values():
    insert_to_path(osp.join('Jacinle/', v['root']))