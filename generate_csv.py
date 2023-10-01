
import glob
import pandas as pd

df = pd.DataFrame(columns=['filepath', 'label'])

dirs = glob.glob('./dataset/*')
files = []

for directory in dirs:
    print(directory)
    files.extend(glob.glob(directory + '/*.png'))

df.filepath = files
df.label = [elem.split('_')[1].split('.')[0] for elem in files]
df.to_csv("ds_info.csv", index=False)