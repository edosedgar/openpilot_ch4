#!/usr/bin/env python3
import os
import sys
import csv
import cv2
import av
import numpy as np

def extract_label_from_filename(dirname):
    # Extracting the label from the end of the directory name
    return dirname.split('--')[-1].split('-')[-1]

def generate_dataset(input_directory, output_directory):
    # Ensure the output directory exists or create it
    os.makedirs(output_directory, exist_ok=True)

    # CSV to store path and label
    csv_filename = os.path.join(output_directory, 'label_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filepath", "Label"])

        # Using os.walk to iterate through the nested directories and files
        for dirpath, dirnames, filenames in os.walk(input_directory):
            for video_file in filenames:
                if video_file == 'ecamera.hevc':
                    video_path = os.path.join(dirpath, video_file)
                    label = extract_label_from_filename(os.path.basename(dirpath))
                    
                    # Creating a directory for the label if it doesn't exist
                    label_dir = os.path.join(output_directory, label)
                    os.makedirs(label_dir, exist_ok=True)

                    container = av.open(video_path)
                    frame_count = 0

                    for frame in container.decode(video=0):
                        img_yuv = frame.to_ndarray(format=av.video.format.VideoFormat('yuv420p'))
                        
                        # Save the image in the labeled directory
                        output_filename = os.path.join(label_dir, f"{video_file.split('.')[0]}_frame{frame_count}.png")
                        cv2.imwrite(output_filename, img_yuv.astype(np.uint8))
                        csv_writer.writerow([output_filename, label])  # Write to CSV
                        frame_count += 1

    print("Dataset generation completed.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: script.py [input_directory] [output_directory]")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    generate_dataset(input_directory, output_directory)
