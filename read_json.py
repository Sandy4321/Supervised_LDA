import sys
import json
import os
import pandas as pd
import numpy as np

#Take dataset folder path as argument
dir_path = sys.argv[1]
print dir_path

content = []

for fil in os.listdir(dir_path):
    fil_path = os.path.join(dir_path, fil)
    print fil_path
    label = fil_path.split('_')[1]
    i=0

    with open(fil_path, 'r') as f:
        for i,line in enumerate(f):
            if i < 10000:
                pass
            elif (i > 10000) and (i < 15000):
                l = json.loads(line)
                content.append([l['reviewText'], label])
            # print json.dumps(l, indent=6, sort_keys=True)
            elif i>15000:
                break

# content_arr = np.array(content)
content_df = pd.DataFrame(content, columns=['Text', 'Label'])
print content_df.head()
print content_df['Label'].value_counts()

f_out = "test.csv"
content_df.to_csv(f_out)
