
#for building application as a package itself
from setuptools import find_packages, setup
# import typing



HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->list[str]:
    """
    This will return a list of requirement
    """
    requirements=[]
    with open (file_path) as file_obj:
        requiremnents=file_obj.readlines()
        requirements=[req.replace('\n', '') for req in requiremnents]


     #this will remove -e from running or showing up in here but run in background 
        if HYPHEN_E_DOT in requiremnents:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='global economy indicator',
    version='0.0.1',
    author='Tiamz',
    author_email='Tiami.abiola@gmail.com',
    package= find_packages(),
    install_require=get_requirements('requirements.txt')

)