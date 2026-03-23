import os
import glob
import xml.etree.ElementTree as ET
import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip

def parse_time(time_str):
    """
    Parses SSBD XML time strings into start and end seconds.
    Cases:
    - '18:24' -> start 18, end 24
    - '0018:0021' -> start 00:18 (18s), end 00:21 (21s)
    - '0130:0148' -> start 01:30 (90s), end 01:48 (108s)
    """
    time_str = time_str.strip()
    if ':' in time_str:
        start_str, end_str = time_str.split(':', 1)
        def to_sec(s):
            s = s.strip()
            if len(s) == 4 and s.isdigit(): # '0130' format
                return int(s[:2])*60 + int(s[2:])
            else:
                return float(s)
        try:
            return to_sec(start_str), to_sec(end_str)
        except:
            return None, None
    return None, None

def download_and_process_ssbd(ssbd_dir):
    annotations_dir = os.path.join(ssbd_dir, "Annotations")
    temp_dir = os.path.join(ssbd_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    
    classes = ["armflapping", "headbanging", "spinning"]
    for c in classes:
        os.makedirs(os.path.join(ssbd_dir, c), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    print(f"Found {len(xml_files)} annotation files.")
    
    ydl_opts = {
        # 'best' downloads a pre-merged video+audio file so ffmpeg isn't required by yt-dlp
        'format': 'best[height<=480][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    successful = 0
    failed = 0
    
    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            url_elem = root.find("url")
            if url_elem is None:
                continue
            video_url = url_elem.text.strip()
            
            print(f"\nProcessing {base_name} ({video_url})")
            
            # Download full video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(video_url, download=True)
                    video_id = info['id']
                    ext = info['ext']
                    downloaded_file = os.path.join(temp_dir, f"{video_id}.{ext}")
                except Exception as e:
                    print(f"  -> Failed to download: {e}")
                    failed += 1
                    continue
            
            # Find behaviours to extract
            behaviours_node = root.find("behaviours")
            if behaviours_node is not None:
                for b_idx, beh in enumerate(behaviours_node.findall("behaviour")):
                    b_id = beh.get("id", f"b_{b_idx}")
                    cat_elem = beh.find("category")
                    time_elem = beh.find("time")
                    
                    if cat_elem is None or time_elem is None: continue
                    
                    category = cat_elem.text.lower().strip()
                    if "arm" in category: folder = "armflapping"
                    elif "head" in category: folder = "headbanging"
                    elif "spin" in category: folder = "spinning"
                    else: continue
                    
                    start_sec, end_sec = parse_time(time_elem.text)
                    if start_sec is None: continue
                    
                    output_file = os.path.join(ssbd_dir, folder, f"{base_name}_{b_id}.mp4")
                    
                    if os.path.exists(output_file):
                        print(f"  -> Skipping {output_file} (already exists)")
                        continue
                        
                    try:
                        print(f"  -> Extracting clip {start_sec}s to {end_sec}s -> {folder}/{os.path.basename(output_file)}")
                        with VideoFileClip(downloaded_file) as video:
                            try:
                                clip = video.subclipped(start_sec, min(end_sec, video.duration))
                            except AttributeError:
                                # Fallback for MoviePy 1.0
                                clip = video.subclip(start_sec, min(end_sec, video.duration))
                                
                            # mute audio completely to avoid codec issues, we only need visual features
                            clip = clip.without_audio()
                            clip.write_videofile(
                                output_file, 
                                codec="libx264", 
                                audio=False, 
                                verbose=False, 
                                logger=None
                            )
                    except Exception as e:
                        print(f"  -> Failed to extract clip: {e}")
            
            # Cleanup downloaded temporary full video
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
            successful += 1
            
        except Exception as e:
            print(f"Error parsing XML {base_name}: {e}")
            failed += 1

    print(f"\nFINISHED. Successfully processed {successful} videos. Failed {failed}.")
    print("Failed videos are usually due to YouTube videos being deleted or made private over time.")

if __name__ == "__main__":
    ssbd_dir = r"D:\WORK\VScode\Capstone\SSBD-file"
    download_and_process_ssbd(ssbd_dir)
