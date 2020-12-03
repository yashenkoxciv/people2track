import cv2
import yaml
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tracker import CentroidTracker


def load_yaml_config(filename):
    ''' This function loads .yml file to Namespace so we can access attributes by name.'''
    with open(filename, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    config = argparse.Namespace(**y)
    return config


def draw_bounding_boxes(frame_image, current_trackables, emotion_time_threshold):
    pil_image = Image.fromarray(frame_image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    for trackable_id in current_trackables:
        trackable = current_trackables[trackable_id]
        top, right, bottom, left = trackable.bounding_boxes[-1]
        # now we've got matches face name (match_face_name)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        # Draw a label with a name below the face
        bounding_box_title = str(trackable_id)
        #text_width, text_height = draw.textsize(bounding_box_title, font=font)
        text_width, text_height = font.getsize(bounding_box_title)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), bounding_box_title, fill=(255, 255, 255, 255), font=font)
    return pil_image


def start_face2know(args, config, tracker):
    '''
    :param args: command-line arguments
    :param config: parsed configuration file
    :param tracker: instance of CentroidTracker
    :return: None
    '''
    if args.video == '0':
        args.video = 0
    video = cv2.VideoCapture(args.video)

    fps_module = int(video.get(cv2.CAP_PROP_FPS) / config.tracking_fps)

    frame_id = -1
    while video.isOpened():
        frame_id += 1
        print('\r', tracker, frame_id + 1, 'frames processed', end='', flush=True)
        do_have_next, frame_image = video.read()
        if not do_have_next:
            break
        if frame_id % fps_module != 0:
            continue
        current_trackables = tracker.update(frame_image)
        if len(current_trackables) == 0:  # didn't find faces
            cv2.imshow('frame', frame_image)
            continue
        # lets match faces and draw bounding boxes around
        pil_image = draw_bounding_boxes(frame_image, current_trackables, config.emotion_time_threshold)
        frame_image = np.array(pil_image)
        cv2.imshow('frame', frame_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('config')
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    tracker = CentroidTracker(config.similarity_threshold, config.max_disappeared)

    start_face2know(args, config, tracker)


