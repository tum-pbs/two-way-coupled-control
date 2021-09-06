import ffmpeg


def stack(streams_list, mode='h'):
    """
    Stack videos horizontally or vertically
        :param streams: list of streams that will be stacked (can be stream or path to videos)
        :param mode: 'h' for horizontal or 'v' for vertical
        :return: stream
    """

    assert(isinstance(streams_list, list))
    assert(mode in ['h', 'v'])

    stream = None
    for input in streams_list:
        if stream is None:

            if type(input) == str: stream = ffmpeg.input(input)
            elif type(input) == ffmpeg.nodes.FilterableStream: stream = input
            else: raise ValueError('streams_list does not contain streams or paths to videos')

        else:
            if type(input) == str: temp_stream = ffmpeg.input(input)
            elif type(input) == ffmpeg.nodes.FilterableStream: temp_stream = input

            stream = ffmpeg.filter_([stream, temp_stream], '%sstack' % (mode))
    return stream


def finish(stream, filename, output_path):
    """
    Run all operations on stream and save video
    :param stream: stream that will be saved
    :param filename: filename of the video
    :param output_path: path to folder that video will be exported to
    """
    stream = stream.output(output_path + filename).run(overwrite_output=True)


def add_text(stream, text, x, y, font_size, anchor='lt', font_path=''):
    """
    Creates a text on top of the stream
    :param stream: stream that will
    :param text: text that will be printed on top of stream
    :param x: x position of text (0 is the left corner)
    :param y: y position of text (0 is the top corner)
    :param font_size: size of font
    :param anchor: where the text is anchored (lt, lm, lb, mt, mm, mb, rt, rm, rb - left, right, top, middle, bottom )
    :param font_path: path to font
    """

    if anchor == 'lt': x_offset, y_offset = '', ''
    elif anchor == 'lm': x_offset, y_offset = '', ' - text_h/2'
    elif anchor == 'lb': x_offset, y_offset = '', ' - text_h'

    elif anchor == 'mt': x_offset, y_offset = '- text_w/2', ''
    elif anchor == 'mm': x_offset, y_offset = '- text_w/2', '-text_h/2'
    elif anchor == 'mb': x_offset, y_offset = '- text_w/2', '-text_h'

    elif anchor == 'rt': x_offset, y_offset = '- text_w', ''
    elif anchor == 'rm': x_offset, y_offset = '- text_w', '-text_h/2'
    elif anchor == 'rb': x_offset, y_offset = '- text_w', '-text_h'

    assert(isinstance(x, str) and isinstance(y, str))
    stream = stream.drawtext(
        text=text,
        x=x + x_offset,
        y=y + y_offset,
        fontfile=font_path,
        fontsize=font_size
    )

    return stream


def overlay_image(stream, image, alpha):
    """
    Overlay an image on top of the stream
    param: stream: stream that will be overlayed
    param: image: imagethat will be overlayed on top of the stream
    param: alpha: transparency of image
    """

    if (isinstance(image, str)): image = ffmpeg.input(image)

    image = image.colorchannelmixer(aa=alpha)
    return ffmpeg.overlay(stream, image)


def images2video(path, images_name, extension, output_name='movie.mp4', framerate=24):
    """
    Convert sequence of images into a video
    :param: path: path to images with part of file names, e.g., path/to/image/cat
    :param: images_name: name of the images
    :param: extension: extension of images, e.g., 'png'
    :param: output_name: name of outputted video file
    :param framerate: outputed video framerate
    """
    stream = ffmpeg.input(path + images_name + '*.' + extension, pattern_type='glob', framerate=framerate)

    finish(stream, output_name, path)


def pad(stream, width, height, x, y, color="white"):
    """
    Pad stream

    Parameters:
        stream: stream to be padded
        width: width of the new stream
        height: height of the new stream
        x: x position of the new stream
        y: y position of the new stream
        color: color of padding

    """
    return ffmpeg.filter_(stream, 'pad', width=width, height=height, x=x, y=y, color=color)
