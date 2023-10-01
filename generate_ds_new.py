#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import paramiko 
import av
import pandas as pd
import tqdm
import glob
from joblib import Parallel, delayed


### To add a new log, simply append to the list of chapters
log_chapters = [
   '2023-09-30--19-47-14--0-R',
   '2023-09-30--19-48-33--0-R',
   '2023-09-30--19-52-27--0-R',
   '2023-09-30--19-53-21--0-R',
   '2023-09-30--19-54-52--0-F',
   '2023-09-30--19-54-52--1-F',
   '2023-09-30--19-56-21--0-F',
   '2023-09-30--19-58-28--0-F',
   '2023-09-30--20-00-02--0-F',
   '2023-09-30--20-00-02--1-F',
   '2023-09-30--20-01-50--0-L',
   '2023-09-30--20-01-50--1-L',
   '2023-09-30--20-02-55--0-L',
   '2023-09-30--20-02-55--1-L',
   '2023-09-30--20-04-05--0-L',
   '2023-09-30--20-04-05--1-L',
   '2023-09-30--22-34-32--0-F',
   '2023-09-30--22-38-48--0-F',
   '2023-09-30--22-39-43--0-F',
   '2023-09-30--22-40-40--0-F',
   '2023-09-30--22-41-56--0-F',
   '2023-09-30--22-41-56--1-F',
   '2023-09-30--22-45-19--0-L',
   '2023-09-30--22-45-19--1-L',
   '2023-09-30--22-47-00--0-R',
   '2023-09-30--22-47-00--1-R',
   '2023-09-30--22-48-43--0-R',
   '2023-09-30--22-49-19--0-R',
   '2023-09-30--22-49-19--1-R',
   '2023-09-30--22-50-37--0-R',
   '2023-10-01--04-29-31--0-R',
   '2023-10-01--04-29-31--1-R',
   '2023-10-01--04-31-59--0-L',
   '2023-10-01--04-31-59--1-L',
   '2023-10-01--04-31-59--2-L',
   '2023-10-01--04-35-15--0-F',
   '2023-10-01--04-35-15--1-F',
   '2023-10-01--11-22-39--0-R',
   '2023-10-01--11-24-10--0-R',
   '2023-10-01--11-24-10--1-R',
   '2023-10-01--11-24-10--2-R',
   '2023-10-01--11-26-39--0-R',
   '2023-10-01--11-31-25--0-L',
   '2023-10-01--11-35-10--0-L',
   '2023-10-01--11-35-10--1-L'
]

storage_path = './dataset_v2/'
df_filename = os.path.join(storage_path, 'desc.csv')
df = pd.DataFrame(columns=['filename', 'label'])

log_chapter_tp = []
### copy
for log_chapter in log_chapters:
   if os.path.isdir(storage_path + '/' + log_chapter) and \
      len(glob.glob(storage_path + '/' + log_chapter + '/*.png')) > 50:
      continue

   log_chapter_tp.append(log_chapter)
   print("Downloading", log_chapter)
   ssh = paramiko.SSHClient()
   ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
   ssh.connect('192.168.63.76', username="comma")
   sftp = ssh.open_sftp()
   localpath = '/tmp/' + log_chapter + '.hevc'
   remotepath = '/data/media/0/realdata/' + log_chapter + '/ecamera.hevc'
   if os.path.isfile(localpath):
      continue
   sftp.get(remotepath, localpath)
   sftp.close()
   ssh.close()

### process
def worker(log_chapter):
   if os.path.isdir(storage_path + '/' + log_chapter) and \
      len(glob.glob(storage_path + '/' + log_chapter + '/*.png')) > 50:
      return

   os.makedirs(storage_path + '/' + log_chapter, exist_ok=True)
   container = av.open(localpath)
   label = log_chapter[-1]
   for frame in container.decode(video=0):
      img_yuv = frame.to_ndarray(format=av.video.format.VideoFormat('yuv420p'))
      h, w = int(img_yuv.shape[0]*2/3), int(img_yuv.shape[1])
      img_yuv = img_yuv[:h,:w]
      img_yuv = img_yuv[:h, 280:-440]
      cv2.imwrite(storage_path + '/' + log_chapter + '/' + str(frame.index) + '.png', img_yuv.astype(np.uint8))
      #df.loc[-1] = [log_chapter + '/' + str(frame.index) + '.png', label]
      #df.index = df.index + 1

   container.close()
   os.remove(localpath)

Parallel(n_jobs=2)(delayed(worker)(log_chapter) for log_chapter in tqdm.tqdm(log_chapter_tp))

for log_chapter in tqdm.tqdm(log_chapters):
   df_local = pd.DataFrame(columns=['filename', 'label'])
   files = glob.glob(storage_path + '/' + log_chapter + '/*.png')
   df_local['filename'] = [log_chapter + '/' + elem.split('/')[-1] for elem in files]
   df_local['label'] = log_chapter[-1]
   df = pd.concat([df, df_local]).reset_index(drop=True)

df.to_csv(df_filename, index=False)
## rsync -avh ./dataset_v2 ekaziak1@10.243.88.21:/home/ekaziak1/train_efbn0/