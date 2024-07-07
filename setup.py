from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_req(file_path:str)->List[str]:
    re=[]
    with open(file_path,'r') as f:
        re=f.read().splitlines()
        if HYPEN_E_DOT in re:
            re.remove(HYPEN_E_DOT)
        return re


setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Bhanu',
    author_email='bhanublez@gmail.com',
    install_requires=get_req('requirement.txt'),
    packages=find_packages()
)