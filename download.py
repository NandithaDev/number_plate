import cv2
import yt_dlp
import numpy as np
import os

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'merge_output_format': 'mp4',  # Ensures the final output is MP4
    'outtmpl': 'vid1.mp4',
    'download_sections': {
        '*': {'start_time': 0, 'end_time': 300}  #
    },
    'cookiesfrombrowser': ('firefox',),
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mp4', #mp4 conversion
    }]
}

url = "https://www.youtube.com/watch?v=7HaJArMDKgI"
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
