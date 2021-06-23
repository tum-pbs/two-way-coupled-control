import ffmpeg 

path = '/home/ramos/work/PhiFlow/demos/myscripts/'
# ffmpeg -i domain.mp4 -i fig.png  -filter_complex "[1:v]format=rgba,colorchannelmixer=aa=0.2[fg];[0][fg]overlay"  test.mp4 -y 

video = ffmpeg.input(path + 'domain.mp4')
image = ffmpeg.input(path + 'fig.png')


image = image.colorchannelmixer(aa="0.2")
stream = ffmpeg.overlay(video, image)

image.output(path+'image_w_alpha.png').run(overwrite_output=True)
stream = stream.output(path + 'test.mp4')
stream.run(overwrite_output=True)