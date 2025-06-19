from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read the video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    #Tracker logic now using the bounding boxes
    tracker = Tracker("models/best.pt")

    object_tracks = tracker.get_object_tracks(video_frames, 
                                              read_from_stub=True, 
                                              stub_path="stubs/track_stubs.pkl")

    #Draw output vid
    ## Draw object tracks
    #print(f"Number of video frames: {len(video_frames)}")
    #print(f"Number of tracked frames: {len(object_tracks['players'])}")

    output_video_frames = tracker.draw_circles(video_frames, object_tracks)

    #Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

    
if __name__ == "__main__":
    main()