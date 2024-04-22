import argparse
from run.perspective import PerspectiveVideoProcessor
from run.normal import NormalVideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Map points in a video and overlay an image.")
    parser.add_argument("--methods", type=str, required=True, help="perspective or normal.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--poster_path", type=str, required=True, help="Where poster are stored.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output. ")

    args = parser.parse_args()
    if args.methods == "normal":
        mapper = NormalVideoProcessor(video_path=args.video_path,
                              poster_path=args.poster_path,
                              output_path=args.output_path)
        mapper.run()
    elif args.methods == "perspective":
        mapper = PerspectiveVideoProcessor(video_path=args.video_path,
                                  poster_path=args.poster_path,
                                  output_path=args.output_path)
        mapper.run()
    else:
        raise ValueError("Invalid method. Choose either 'normal' or 'perspective'.")


if __name__ == "__main__":
    main()
