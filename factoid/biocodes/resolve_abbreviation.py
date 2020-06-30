import subprocess

import os
import pdb
from pathlib import Path

current_folder = os.path.dirname(os.path.abspath(__file__))

def resolve_abbreviation(corpus_path):
    folder = Path(current_folder + '/../../Ab3P/identify_abbr').resolve()
    corpus_path = Path(corpus_path).resolve()

    # change working directory to run Ab3P properly
    original_cwd = os.getcwd()
    os.chdir(current_folder)
    result = subprocess.run([str(folder), str(corpus_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # return to origianl working directory
    os.chdir(original_cwd)
    
    line = result.stdout.decode('utf-8')
    error = result.stderr.decode('utf-8')
    if error == "Path file for type cshset does not exist!":
        raise "Path file for type cshset does not exist!"
    elif "Cannot open" in error:
        raise "Cannot open file"
    lines = line.split("\n")
    result = {}
    for line in lines:
        if len(line.split("|"))==3:
            sf, lf, _ = line.split("|")
            sf = sf.strip()
            lf = lf.strip()
            result[sf] = lf
            
    return result ,error

if __name__ == '__main__':
    corpus_path = current_folder + '/23864035.txt'
    print("corpus_path=",corpus_path)
    result = resolve_abbreviation(corpus_path)
    print("result=",result)