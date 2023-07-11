from sys import platform
import os 
global root_path
if platform == "linux":
    root_path =  os.getcwd()
elif platform == "win32" or platform == 'darwin':
    root_path = 'modules/app/src'
