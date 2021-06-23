import ffmpeg 

import os 
import ffmpeg 
import sys 
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)
import ffmpeg_functions as fff



positions = ['10', '15', '20', '25']



for i, x0 in enumerate(positions): 

    row = None 
    for y0 in positions: 

        stream = ffmpeg.input(directory+ '/x%s_y%s/domain.mp4' % (x0, y0))
        stream = fff.add_text(stream, 'x0 = %s, y0 = %s' % (x0, y0), 'w/2', '10', 40, anchor = 'mt' , font_path=  "/home/ramos/.local/share/fonts/Unknown Vendor/TrueType/CMU Serif/CMU_Serif_Roman.ttf")
        
        image_overlay = ffmpeg.input(directory + '/x%s_y%s/domain_target.png' % (x0, y0))
        stream = fff.overlay_image(stream, image_overlay, 0.2)
        row = stream if row is None else fff.stack([row, stream], 'h')

    stack = row if i == 0 else fff.stack([stack,row],'v')


# result = ffmpeg.output(stack, 'test.mp4')
# ffmpeg.view(result)
fff.finish(stack, '/comparison.mp4', directory)

