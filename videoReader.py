import cv2
import sys
import time
def read_video(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    print('Video Frame width({})/height({})'.format(frame_width, frame_height))
    resized_w = int(frame_width / 4)
    resized_h = int(frame_height / 4)
    while ret and cap.isOpened():
        resized_frame = cv2.resize(frame, (resized_w, resized_h))
        cv2.imshow('frame', resized_frame)
        ret, frame = cap.read()
        if cv2.waitKey(13) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    if len(sys.argv) >= 1:
        video_name = sys.argv[1]
        read_video(video_name)