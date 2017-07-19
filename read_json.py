import sys
import json
import os
from random import random
import pandas as pd
import numpy as np

def convert_json_to_csv(path_to_json_files, number_of_lines=10000, output_file='test_new.py'):

    content = []

    for fil in os.listdir(path_to_json_files):
        fil_path = os.path.join(path_to_json_files, fil)
        print fil_path

        label = fil_path.split('_')[1]
        n=0

        with open(fil_path, 'r') as f:

            for line in f:

                if (random() < 0.1) and (n < int(number_of_lines)/2):
                    l = json.loads(line)
                    content.append([l['reviewText'], label])
                    n += 1

                elif (n >= int(number_of_lines)/2):
                    break

                else:
                    pass

    # content_arr = np.array(content)
    content_df = pd.DataFrame(content, columns=['Text', 'Label'])
    print content_df.head()
    print content_df['Label'].value_counts()

    content_df.to_csv(output_file)


convert_json_to_csv(sys.argv[1], sys.argv[2], sys.argv[3])
