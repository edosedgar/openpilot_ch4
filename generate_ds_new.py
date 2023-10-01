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
   '2023-09-30--20-04-05--1-L'
]

storage_path = './dataset_v2/'
df_filename = os.path.join(storage_path, 'desc.csv')

if os.path.isfile(df_filename):
   df = pd.read_csv(df_filename)
else:
   df = pd.DataFrame(columns=['filename', 'label'])

for log_chapter in tqdm.tqdm(log_chapters):
   if os.path.isdir(storage_path + '/' + log_chapter) and \
      len(glob.glob(storage_path + '/' + log_chapter + '/*.png')) > 5:
      continue
   
   ssh = paramiko.SSHClient()
   ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
   ssh.connect('192.168.63.76', username="comma")
   sftp = ssh.open_sftp()
   localpath = '/tmp/' + log_chapter + '.hevc'
   remotepath = '/data/media/0/realdata/' + log_chapter + '/ecamera.hevc'
   sftp.get(remotepath, localpath)
   sftp.close()
   ssh.close()

   os.makedirs(storage_path + '/' + log_chapter, exist_ok=True)
   container = av.open(localpath)
   label = log_chapter[-1]
   for frame in container.decode(video=0):
      img_yuv = frame.to_ndarray(format=av.video.format.VideoFormat('yuv420p'))
      h, w = int(img_yuv.shape[0]*2/3), int(img_yuv.shape[1])
      img_yuv = img_yuv[:h,:w]
      cv2.imwrite(storage_path + '/' + log_chapter + '/' + str(frame.index) + '.png', img_yuv.astype(np.uint8))
      df.loc[-1] = [log_chapter + '/' + str(frame.index) + '.png', label]
      df.index = df.index + 1

   container.close()
   os.remove(localpath)
   df.to_csv(df_filename, index=False)