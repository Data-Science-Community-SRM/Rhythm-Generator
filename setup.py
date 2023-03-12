from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        #it will read all the lines in requirements.txt file including the '\n'
        #to replace the '/n' with the blank
        requirements=[req.replace("\n","")for req in requirements]

        #'-e .' in the requirements.txt file use to trigger the setup.py
        # but '-e .' will also get read
        # to remove this

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) 
    return requirements


setup(
    name = 'Rhythm Generator',
    version= '0.0.1',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")


)