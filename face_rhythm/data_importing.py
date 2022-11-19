class Dataset_videos:
    def __init__(self, path, frame_rate, video_names, video_lengths):
        self.path = path
        self.frame_rate = frame_rate
        self.video_names = video_names
        self.video_lengths = video_lengths

    def get_video(self, video_name):
        return Video(self.path, self.frame_rate, video_name, self.video_lengths[video_name])