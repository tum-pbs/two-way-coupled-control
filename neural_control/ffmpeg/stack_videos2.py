import ffmpeg
import ffmpeg_functions as fff


path = "/home/ramos/phiflow/storage/controller/movies/"
videos = {
    "re3000": {
        "filename": "pid.mp4",
        "title": 'PID'
    },
    # "re8000": {
    #     "filename": "2obs_re8000.mp4",
    #     "title": 'Re 8000'
    # },
}

streams = []
# Crop videos
isFirst = True
for video in videos.values():
    stream1 = (ffmpeg
               .input(path + video["filename"])
               .crop(0, 0, "iw/2", "ih/3")
               )
    stream2 = (ffmpeg
               .input(path + video["filename"])
               .crop("iw/2", "ih/3", "iw/2", "ih/3")
               )
    stream3 = (ffmpeg
               .input(path + video["filename"])
               .crop("iw/2", "2*ih/3", "iw/2", "ih/3")
               )
    stream = fff.stack([stream1, stream2, stream3])
    if isFirst:
        stream = fff.pad(stream, "iw", "ih+10", 0, 0, color="black")
        isFirst = False
    stream = fff.pad(stream, "iw", "ih*1.1", 0, "0.1*ih")
    stream = fff.add_text(stream, video["title"], 'w/2-text_w/2', '20', 60, font_path='/Windows/fonts/cmunrm.ttf')
    streams += [stream]


stream = fff.stack(streams, mode="v")
stream = (ffmpeg
          .output(stream, path + "out.mp4")
          .run(overwrite_output=True)
          )
