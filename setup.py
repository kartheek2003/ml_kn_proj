from setuptools import setup , find_packages

from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str)->List[str]:
    '''Returns a list of requirements'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [line.replace("\n","") for line in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements






setup(
    name='ML_KN_PROJ',
    version='0.0.1',
    author = "kartheek",
    author_email= "kartheekakkabathula@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') 
    
)