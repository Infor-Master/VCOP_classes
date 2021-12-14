import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard

def capture_videos(filename):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    video_ready = False
    writerC = None
    writerD = None

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if video_ready == False:
                depthSize = (int(depth_frame.get_width()), int(depth_frame.get_height()))
                colorSize = (int(color_frame.get_width()), int(color_frame.get_height()))
                fps = color_frame.get_profile().fps()  # color and depth have the same fps
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writerD = cv2.VideoWriter("video_depth_"+filename+".mov", fourcc, fps, depthSize, isColor=False)
                writerC = cv2.VideoWriter("video_color_"+filename+".mov", fourcc, fps, colorSize, isColor=True)
                video_ready = True

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_image_grayscale_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            writerD.write(depth_image_grayscale_8bit)
            writerC.write(color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        # Stop streaming
        pipeline.stop()
        writerC.release()
        writerD.release()
        cv2.destroyAllWindows()

count = 0
while True:
    if keyboard.is_pressed('<'):
        capture_videos("escada_001"+str(count))
        count +=1