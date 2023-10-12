import ffmpeg
import ffmpeg_functions as fff


path = "/home/ramos/phiflow/figs/"
videos = {
    "PID": {
        "filename": "pid.mp4",
        "title": 'PID'
    },
    # "loop_shaped": {
    #     "filename": "loop_shaped.mp4",
    #     "title": 'Loop Shaped'
    # },
    # "supervised": {
    #     "filename": "supervised.mp4",
    #     "title": 'Supervised'
    # },
    # "felix": {
    #     "filename": "rl.mp4",
    #     "title": 'RL'
    # },
    "online": {
        "filename": "online.mp4",
        "title": 'Online'
    }
    # "smoke": {
    #     "filename": "smoke.mp4",
    #     "title": 'Online'
    # },

}

streams = []
# Crop videos
isFirst = True
for video in videos.values():
    stream = (ffmpeg
              .input(path + video["filename"])
              #   .crop(0, "0", "iw/2", "ih/3")
              .crop(120, "200", "iw*0.8", "ih*0.6")

              )
    # if isFirst:
    #     stream = fff.pad(stream, "iw", "ih+10", 0, 0, color="black")
    #     isFirst = False
    stream = fff.pad(stream, "iw", "ih*1.1", 0, "0.1*ih")
    stream = fff.add_text(stream, video["title"], 'w/2-text_w/2', '20', 60, font_path='/Windows/fonts/cmunrm.ttf')
    streams += [stream]


stream = fff.stack(streams, mode="h")
stream = (ffmpeg
          .output(stream, path + "out.mp4")
          .run(overwrite_output=True)
          )
