import os.path
import shutil
from datetime import datetime


def dir_gen(base, filename):
    subdir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    script_name = filename.split('/')[-1]
    script_name = script_name.split('.')[0]
    subdir = subdir + '-' + script_name
    directory = os.path.join(base, subdir)
    if not os.path.isdir(directory):  # Create the log directory if it doesn't exist
        os.makedirs(directory)
    shutil.copy(filename, directory)
    return directory
