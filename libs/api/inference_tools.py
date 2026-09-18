#%%
import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
import albumentations as A
from mmdet.apis import init_detector
from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp
from libs.utils.visualizer import visualize_lanes
from tqdm import tqdm
import shutil
import natsort

#%%
def decompose_video(video_path, output_path):
    """
    video_path: video file path
    output_path: save path for video frames
    """
    print("input video for decomposition: ", video_path)
    print("decomposed frame save path: ", output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    print("frame shape (h, w, c): ", image.shape)    

    while success:
        cv2.imwrite(os.path.join(output_path, "%06d.jpg" % count), image)
        success, image = vidcap.read()
        count += 1
        if count % 10000 == 0:
            print(f"{count // 10000} * 10000 th frames counted!")
    print("Finished reading video!")
    
def compose_video(frame_path, output_path):
    """
    frame_path: folder that contains the frames from video.
    output_path: output path that saves the composed video.
    """
    print("input frame folder for composition: ", frame_path)
    print("output folder for saving composed video: ", output_path)
    # extract directory path except the filename
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    videos = os.listdir(output_path)
    name_count = len([video for video in videos if video.endswith(('.mov', 'mp4'))])
    
    frames = os.listdir(frame_path)
    frames = natsort.natsorted(frames) # order the frames as their natural ordering

    if len(frames) == 0:
        print("Error: There is no any frame to compose!")
        exit()

    init_frame = cv2.imread(os.path.join(frame_path, frames[0]))
    frame_h = init_frame.shape[0]
    frame_w = init_frame.shape[1]
    # define the codec and create videowriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(os.path.join(output_path, f"video_{name_count}.mp4"), fourcc, 60.0, (frame_w, frame_h))

    for frame in tqdm(frames, desc="composing frames"):
        frame_img = cv2.imread(os.path.join(frame_path, frame))
        out.write(frame_img)

    out.release() # release the memory for video writer
    cv2.destroyAllWindows()

def webcam_inference(argument):
    cap = cv2.VideoCapture(argument.webcam_num)

    if not cap.isOpened():
        print("Error: Could not access the webcam")
        return
    else:
        print("Webcam accessed successfully!")

    # load the trained weights to the detector
    detector = load_models(argument)
    # build the data pipeline
    test_pipeline = Compose(detector.cfg.test_dataloader.dataset.pipeline)

    # frame calculation
    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX # - set the font for fps

    # Read the first frame to confirm capturing
    ret, frame = cap.read()
    print("frame shape (h,w,c): ", frame.shape)

    if argument.save_and_compose:
        if not os.path.exists(argument.stream_save_path):
            os.makedirs(argument.stream_save_path)
        if not os.path.exists(argument.stream_video_path):
            os.makedirs(argument.stream_video_path)
        count = 0

    while ret:

        # data processing
        data = dict(filename="", sub_img_name=None, img=frame, gt_points=[], id_classes=[], id_instances=[], img_shape=frame.shape, ori_shape=frame.shape)
        data = test_pipeline(data)
        data_ = dict(
            inputs=[data["inputs"]],
            data_samples=[data["data_samples"]],
        )

        # forward the model
        with torch.no_grad():
            results = detector.test_step(data_)
        
        # retrieve the inference result and visualize
        lanes = results[0]['lanes']
        ori_h = data['data_samples'].ori_shape[0]
        ori_w = data['data_samples'].ori_shape[1]

        for lane in lanes:
            lane = lane.cpu().numpy()
            xs = lane[:, 0]
            ys = lane[:, 1]
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
            interp_pred = interp(pred, p=5)

        vis = visualize_lanes(frame, interp_pred, save_path=None)

        if argument.save_and_compose:
            # currently support up to 999,999 frame savings (about 4.5 hours of video using 60 frame encoding)
            cv2.imwrite(os.path.join(argument.stream_save_path, "%06d.jpg" % count), vis)
            count = count + 1

        # calculate the FPS (Frame per Second)
        new_frame_time = time.time() # - when the processing this frame is finished
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = "%.2f" % fps        
        cv2.putText(vis, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("predicted lane visualization", vis)

        # wait for the keyboard for 1ms
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'): # exit on ESC or 'q'
            break

        # read the next frame
        ret, frame = cap.read()
    
    # finished reading the frame
    cv2.destroyAllWindows()
    cap.release()

    if argument.save_and_compose:
        # compose the video
        compose_video(argument.stream_save_path, argument.stream_video_path)



#%%
# Path to the repository

WORK_DIR = '/work/CLRerNet'

def arg_parse():
    
    def nullable_string(val):
        if val.lower() in ('none', ''):
            return None
        return val
    
    def boolean_string(val):
        if val.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif val.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description="Camera Real-time Inference")
    parser.add_argument("--model_path", type=str, help="Path to the model weights")
    parser.add_argument("--img_path", type=str, help="Path to the image or directory of images")
    parser.add_argument("--config_path", type=str, help="Path to the model config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold for displaying lanes")    
    parser.add_argument("--out_dir", type=nullable_string, default=None, help="Directory to save output images. If None, images will not be saved. If video_mode=True, this will be the absolute path to save videos")

    parser.add_argument("--video_mode", type=boolean_string, default=False, help="Set to True if input is a video file and made a video from the output images")
    parser.add_argument("--decompose", type=boolean_string, default=False, help="Only valid whe video_mode=True, Set the option whether to decompose the given video")
    parser.add_argument("--compose", type=boolean_string, default=False, help="Only valid when video_mode=True, set whether to compose the video from the output images of frames")
    parser.add_argument("--inference", type=boolean_string, default=False, help="Only valid when video_mode=True, set whether inference on the list of images")
    parser.add_argument("--decompose_video_path", type=str, default="", help="Only valid when video_mode=True, decompose=True, set path of video to decompose")
    parser.add_argument("--decompose_out_path", type=str, default="", help="Only valid when video_mode=True, decompose=True, Output path for saving frames")
    parser.add_argument("--video_out_path", type=str, default="", help="Only valid when video_mode=True, compose=True, set the video output path")
    parser.add_argument("--delete_imgs", type=boolean_string, default=True, help="Only valid when video_mode=True, compose=True, whether to delete the inference frames after composing the video")

    parser.add_argument("--camera_mode", type=boolean_string, default=False, help="Set to True, when you want to inference in real-time from a camera stream")
    parser.add_argument("--webcam_num", type=int, default=0, help="source number of camera (default as 0, as it is the default value of webcam camera)")
    parser.add_argument("--save_and_compose", type=boolean_string, default=False, help="Whether to save the frame from the camera stream and compose the video from the saved frames")
    parser.add_argument("--stream_save_path", type=str, default='', help='Only valid when camera_mode=True, save_and_compose=True, save path for frames from camera stream')
    parser.add_argument("--stream_video_path", type=str, default='', help='Only valid when camera_mode=True, save_and_compose=True, save path for composed video')

    return parser.parse_args()

    
def load_images(argument):
    argument = arg_parse()
    img_path = argument.img_path
    print(f"Image_path: {img_path}")
    
    if os.path.isdir(img_path):
        img_list = [os.path.join(img_path, img_name) for img_name in os.listdir(img_path) if img_name.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        img_list = [img_path]
    
    if len(img_list) == 0:
        raise Exception("Cannot find any images!")
    
    dataset = []
    imgs = []
    for i, path in enumerate(tqdm(img_list, desc="loading images ...")):
        img = cv2.imread(path)
        ori_shape = img.shape
        imgs.append(img)
        data = dict(
            filename=path,
            sub_img_name=None,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        dataset.append(data)

    return imgs, dataset

def load_models(argument):
    model_path = argument.model_path
    config_path = argument.config_path
    device = argument.device
    print(f"Model_path: {model_path}")
    print(f"Config_path: {config_path}")
    print(f"Device: {device}")
    
    model = init_detector(config_path, model_path, device=device)
    model.bbox_head.test_cfg.as_lanes = False
    
    return model

def inference(model, dataset):
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    
    predictions = []
    for i, data in enumerate(tqdm(dataset, desc="inference on images")):
        data = test_pipeline(data) # apply data processing (defined in model configuration file)
        data_ = dict(
            inputs=[data["inputs"]],
            data_samples=[data["data_samples"]],
        )
        
        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)
        
        lanes = results[0]['lanes']
        ori_h = data['data_samples'].ori_shape[0]
        ori_w = data['data_samples'].ori_shape[1]
        # preds = get_prediction(lanes, data['ori_shape'][0], data['ori_shape'][1])
        
        preds = []
        for lane in lanes:
            lane = lane.cpu().numpy()
            xs = lane[:, 0]
            ys = lane[:, 1]
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
            interp_pred = interp(pred, n=5)
            preds.append(interp_pred)
        predictions.append(preds)
    return predictions
            
def visualize(predictions, imgs, dataset, arguments):
    
    vis_out = []
    # Visualize and optionally save the output images
    for i, preds in enumerate(tqdm(predictions, desc="visualizing images")):
        if arguments.out_dir is not None:
            token = dataset[i]['filename'].split('/')[-1].split('.')
            filename = token[-2]
            ext = token[-1]
            save_path = os.path.join(arguments.out_dir, f"{filename}_out.{ext}")
        else:
            save_path = None
        src = imgs[i]
        dst = visualize_lanes(src, preds, save_path=save_path) # output image
        vis_out.append(dst)    
    return vis_out
                

def inference_pipeline(argument):

    argument.model_path = os.path.join(WORK_DIR, argument.model_path)
    argument.config_path = os.path.join(WORK_DIR, argument.config_path)
    
    imgs, dataset = load_images(argument)
    model = load_models(argument)
    predictions = inference(model, dataset)
    
    # if out_dir is specified and not exists, create it
    if argument.out_dir is not None and not os.path.exists(argument.out_dir):
        os.makedirs(argument.out_dir, exist_ok=True)
    
    vis = visualize(predictions, imgs, dataset, argument)

# %%
def main():
    argument = arg_parse()

    if argument.camera_mode:
        argument.img_path = None # if camera_mode=True, img_path will not be set.
        webcam_inference(argument)
        return

    if argument.video_mode:
        # image path for inference is same as decompose output path
        argument.img_path = argument.decompose_out_path
        if argument.decompose:
            decompose_video(argument.decompose_video_path, argument.decompose_out_path)
        if argument.inference:
            inference_pipeline(argument)
        if argument.compose:
            compose_video(argument.out_dir, argument.video_out_path)
            if argument.delete_imgs:
                shutil.rmtree(argument.out_dir, ignore_errors=True)
    else:
        inference_pipeline(argument)

if __name__ == "__main__":
    main()