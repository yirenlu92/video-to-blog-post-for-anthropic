import modal
import cv2
import numpy as np
import argparse


image = (
    modal.Image.debian_slim()
    .pip_install(
        "pytube",
        "pydub",
        "numpy",
        "requests",
        "pyjwt",
        "boto3",
        "pyairtable",
        "opencv-python",
        "numpy",
        "openai",
        "assemblyai",
        "anthropic",
        "pybase64",
        "httpx",
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)

app = modal.App("video-to-blog-post", image=image)


def extract_blog_post_section(text):
    import re

    # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    pattern = r"<blog_post_section>(.*?)</blog_post_section>"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return text


def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        first_line_of_item = item.strip().split("\n")[0]
        if first_line_of_item not in seen:
            seen.add(first_line_of_item)
            result.append(item)
    return result


def extract_slide_text(text):
    import re

    # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    pattern = r"<slide_text>(.*?)</slide_text>"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        return remove_duplicates_preserve_order(results)
    else:
        return [text]


def upload_file_to_r2(file_path):
    import boto3
    import json

    credentials = {}

    # Load credentials from the JSON file
    with open("path/to/credentials.json") as f:
        credentials = json.load(f)

    # Extract credentials
    aws_access_key_id = credentials.get("aws_access_key_id")
    aws_secret_access_key = credentials.get("aws_secret_access_key")

    s3 = boto3.client("s3")

    s3 = boto3.client(
        service_name="s3",
        endpoint_url="https://36cc38112bef9dac3e0dce835950cd6e.r2.cloudflarestorage.com",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
    )

    # Upload/Update single file
    s3.upload_file(
        file_path,
        Bucket="video-to-blog-post-uploads",
        Key=file_path,
    )

    # get the public url from r2
    public_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{file_path}"

    print(f"public_url: {public_url}")
    return public_url


class YoutubeVideo:
    def __init__(self, url):
        self.url = url

    def download_youtube_audio(self, url):
        from pytube import YouTube
        import re

        # Download video from YouTube
        yt = YouTube(url)
        self.title = yt.title
        self.description = yt.description

        video = yt.streams.first()
        # lower-case the title of the youtube video, remove punctuation and replace spaces with underscores
        file_name = re.sub(r"[^\w\s]", "", self.title.lower()).replace(" ", "_")

        audio_file = video.download(filename=f"{file_name}.mp4")

        # upload the file to s3
        audio_url = upload_file_to_r2(audio_file)

        return audio_url


@app.function(timeout=6000)
def read_text_from_slides_with_anthropic(slides_start, slides_end):
    # get the base64 string for each image
    import base64
    import httpx

    import anthropic

    image_text_prompt = """
You are an AI assistant tasked with analyzing a series of images containing presentation slides. Your job is to extract and structure the text from these slides, as well as render any diagrams present. Follow these instructions carefully for each image:

For each image, numbered in ascending order, please do the following:

1. Perform Optical Character Recognition (OCR) on the text in the slide portion of the image. Ignore any text in other sections of the image, such as titles or parts showing the speaker.

2. Structure the OCR'ed text to resemble its appearance on the slide as closely as possible. Present this text within <slide_text> tags.

3. If the slide contains any diagrams, render them as best you can.

4. Output the results for each slide in ascending order based on the image numbers. Format your output as follows:

<image_analysis number="1">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>
</image_analysis>

<image_analysis number="2">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>
</image_analysis>

[Continue for all images in the set]

Example of how to structure the OCR'ed text:

<slide_text>
Title of Slide

• Bullet point 1
• Bullet point 2
   - Sub-bullet point A
   - Sub-bullet point B
• Bullet point 3

Additional text on the slide
</slide_text>


Remember to maintain the exact order of the image numbering in your analysis and to include all relevant information from each slide.
    """

    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]

    for i in range(slides_start, slides_end):
        image_url = (
            f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/slides/slide_{i}.png"
        )
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        image_media_type = "image/png"

        messages[0]["content"].append(
            {"type": "text", "text": f"Image {i}:"},
        )
        messages[0]["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_data,
                },
            }
        )

    messages[0]["content"].append({"type": "text", "text": image_text_prompt})

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=2000, messages=messages
        )
        print(message.content[0].text)
        return message.content[0].text

    except Exception as e:
        print(f"Error in Anthropic API: {e}")
        return f"Error in Anthropic API: {e}"


@app.function(timeout=6000)
def write_section(transcript, slide_text):
    import anthropic

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
    )

    prompt = f"""The following is the transcript of a video recording of a conference talk, and an OCR'ed slide from the talk. Please do the following:
    
    1. pull out the portion of the transcript that corresponds to the slide text
    2. lightly edit that portion of the transcript for clarity. The edit should preserve the first person voice of the talk, while converting it into a segment of a techncial blog post.
       a. The section should contain all the detail of the transcript segment and the slide text, but in a more polished and readable form with grammatically correct sentences.
       b. The section should have the same subheading as the slide title, in H2 format, in sentence case.
       c. The section should incorporate the information from the corresponding slide. When appropriate, it should include the slide text verbatim.
       d. Output the section, in markdown format, in <blog_post_section> tags

    <slide_text>
    {slide_text}
    </slide_text>

    Here is the transcript:
    <transcript>
    {transcript}
    </transcript>
    """

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0.0,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )

        print(message.content[0].text)
        return extract_blog_post_section(message.content[0].text)
    except Exception as e:
        print(f"Error in Claude API: {e}")
        return f"Error in Claude API: {e}"


def transcribe_with_assembly(
    audio_url=None,
):
    import os

    # Make call to Assembly AI to transcribe with speaker labels and
    import assemblyai as aai

    aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcript = transcriber.transcribe(audio_url, config)

    return transcript.text


def save_slides_from_video(
    video_path, output_folder, frame_interval=100, threshold=0.3
):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    slide_count = 0
    prev_frame = None
    slides = []

    slide_ms_timestamp_start = 0
    slide_ms_timestamp_end_prev = 0
    slide_path = ""
    image_urls = []

    for i in range(0, frame_count, frame_interval):
        # Set video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read frame
        ret, current_frame = cap.read()
        if not ret:
            break

        # if this is the first frame, save it as the first slide
        if prev_frame is None:
            # save the first frame
            slide_path = f"{output_folder}/slide_{slide_count}.png"
            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

            image_url = upload_file_to_r2(slide_path)

        # if this is not the first frame, compare it to the previous frame
        if prev_frame is not None:
            # Calculate the difference between frames
            diff = cv2.absdiff(current_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)

            # If difference is significant, it's a new slide
            if non_zero_count > threshold * diff.size:
                # Record the timestamp of the slide
                timestamp = i / fps
                print(f"New slide at {timestamp} seconds.")

                ms_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                # end of the previous slide was just before the start of the current slide
                slide_ms_timestamp_end_prev = ms_timestamp - 100

                # save the previous slide
                slides.append(
                    (
                        slide_count,
                        slide_ms_timestamp_start,
                        slide_ms_timestamp_end_prev,
                        slide_path,
                    )
                )

                # start the next slide
                slide_ms_timestamp_start = ms_timestamp

                # increment the slide count
                slide_count += 1

                # get the slide_path
                slide_path = f"{output_folder}/slide_{slide_count}.png"

                # add a thin black rectangle to the bottom of the image with white text with the image path
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_black = np.zeros((100, current_frame.shape[1], 3), np.uint8)
                cv2.putText(
                    bottom_black,
                    f"image_path: {slide_path}",
                    (10, 70),
                    font,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(slide_path, current_frame)
                print(f"Slide {slide_count} saved.")

                # upload the slide to r2
                image_url = upload_file_to_r2(slide_path)

        # Update previous frame
        prev_frame = current_frame

    cap.release()
    print("Finished processing video.")
    return slide_count


def create_video_to_post(youtube_url):
    # get the youtube video and other assorted metadata
    video = YoutubeVideo(youtube_url)

    # download the youtube video
    video_url = video.download_youtube_audio(youtube_url)

    # get the video_url
    print(video_url)

    length_slides = save_slides_from_video(
        video_path=video_url,
        output_folder="slides",
        frame_interval=100,
        threshold=0.3,
    )

    # split the slides into batches of 20

    read_text_from_slides_with_anthropic_f = modal.Function.lookup(
        "video-to-blog-post", "read_text_from_slides_with_anthropic"
    )

    slide_text_list = []
    for x in read_text_from_slides_with_anthropic_f.starmap(
        [(x, x + 10) for x in range(1, length_slides, 10)]
    ):
        slide_text_list.append(x)

    slide_text_list = [extract_slide_text(x) for x in slide_text_list if x is not None]

    # flatten list

    slide_text_list = [item for sublist in slide_text_list for item in sublist]

    # check the first line of each item in the slide_text_list and dedupe based on that
    slide_text_list = remove_duplicates_preserve_order(slide_text_list)

    with open("slide_text_list.txt", "w") as f:
        for item in slide_text_list:
            f.write("%s\n" % item)

    # # transcribe the video
    transcript = transcribe_with_assembly(audio_url=video_url)

    write_section_f = modal.Function.lookup("video-to-blog-post", "write_section")

    # for each slide text, grab the corresponding portion of the transcript, and rewrite it into a section of the blog post
    written_sections = write_section_f.starmap(
        [(transcript, x) for x in slide_text_list]
    )

    with open("blog_post.md", "w") as f:
        f.write("\n".join(written_sections))


def main():
    parser = argparse.ArgumentParser(description="Convert Youtube video to blog post")

    parser.add_argument(
        "--youtube_url",
        type=str,
        required=True,
        help="Url of the Youtube video that you would like to convert to a blog post",
    )
    args = parser.parse_args()

    # Submit the job to Modal
    create_video_to_post(args.youtube_url)


if __name__ == "__main__":
    main()
