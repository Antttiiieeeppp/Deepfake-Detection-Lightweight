from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# apply when the result is a value
def adding_subtitles_to_video(video_file):
    # get input video
    video = VideoFileClip(video_file)

    # Create a text clip for the subtitle
    subtitle = TextClip("Deepfake", fontsize=30, color='red', font="Arial-Bold")
    subtitle = subtitle.set_pos('right', 'top').set_duration(video.duration)

    video_with_subtitle = CompositeVideoClip([video, subtitle])

    temp_output = "/path/to/output_video.mp4"
    video_with_subtitle.write_videofile(temp_output, codec="libx264")

    return temp_output


# apply when the result is a list
def adding_subtitles_to_video_w_list(video_file, result_list):
    video = VideoFileClip(video_file)

    subtitle_clips = []

    # timestamp = deepfake 일때만 나오게
    for i, detection in enumerate(result_list):
        if detection == 1:  
            start_time = i * (video.duration / len(result_list))
            subtitle = TextClip("Deepfake", fontsize=30, color='red', font="Arial-Bold")
            subtitle = subtitle.set_pos(('right', 'top')).set_start(start_time).set_duration(video.duration / len(result_list))
            subtitle_clips.append(subtitle)

    video_with_subtitles = CompositeVideoClip([video] + subtitle_clips)

    temp_output = "/path/to/output_video.mp4"
    video_with_subtitles.write_videofile(temp_output, codec="libx264")

    return temp_output


# apply this if the result is a dictionary
def adding_subtitles_to_video_w_dict(video_file, result_dict):
    video = VideoFileClip(video_file)
    # Initialize an empty list to hold subtitle clips
    subtitle_clips = []

    for timestamp, detection in result_dict.items():
        if detection == 1:  # deepfake,
            subtitle = TextClip("Deepfake", fontsize=30, color='red', font="Arial-Bold")
            subtitle = subtitle.set_pos(('right', 'top')).set_start(timestamp).set_duration(2)  # -> 2 seconds duration for subtitle
            subtitle_clips.append(subtitle)

    video_with_subtitles = CompositeVideoClip([video] + subtitle_clips)

    # Save the modified video to a temporary dir
    temp_output = "/path/to/output_video.mp4"
    video_with_subtitles.write_videofile(temp_output, codec="libx264")

    return temp_output

