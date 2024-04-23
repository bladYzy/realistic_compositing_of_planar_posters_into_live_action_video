
process videos by mapping points in the video and overlaying images using two different methods: normal and perspective.

#before running the code...
download pretrained models of omnidata and place them in the folder - "\run\pretrained_models"
sh run\tools\download_depth_models.sh


# Usage

python main.py --methods [method] --video_path [path_to_video] --poster_path [path_to_poster] --output_path [path_to_output]

# Example

python video_processor.py --methods normal --video_path /path/to/video.mp4 --poster_path /path/to/poster --output_path /path/to/output.mp4

