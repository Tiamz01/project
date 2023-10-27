from setuptools import find_package, setup
import os
from typing import List



hypen_e_dot = "-e ."
def get_requirement(file_path:str) -> List[str]:

    requirements = []

    with open (file_path) as file_obj:
        requirements = file_obj.readlines() 
        requirements=[requirements.replace("\n", "") for req in requirements]
        
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot) 

    return requirements



setup(
    name='Absenteeism Predictor',
    version=0.0.1',
    author= "Tiamz",
    author_email="Tiami.abiola@gmail.com"
    packages= find_package
    install_require = get_requirement("requirements.txt")
)