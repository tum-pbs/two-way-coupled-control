import ffmpeg
import ffmpeg_functions as fff


path = "C:\\OneDrive\\Brener\\Germany\\TUM\\work\\PhiFlow\\storage\\movies\\"
videos = {
    "300x120": {
        "filename": "300x120.mp4",
        "title": '300 x 120'
    },
    "175 x 110": {
        "filename": "175x110.mp4",
        "title": '175 x 110'
    },
    "175 x 110 (Sponge)": {
        "filename": "175x110_sponge.mp4",
        "title": '175 x 110 (Sponge)'
    }
}

streams = []
# Crop videos
isFirst = True
for video in videos.values():
    stream = (ffmpeg
              .input(path + video["filename"])
              .crop(0, "ih/3", "iw/2", "ih/3")

              )
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
