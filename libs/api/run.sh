model_path="work_dirs/clrernet_culane_segman/run2/epoch_15.pth"
config_path="work_dirs/clrernet_culane_segman/run2/korean_highway_noon_inference_config.py"
img_path="/work/CLRerNet/demo/korean_highway_noon/decomposed"
out_dir="/work/CLRerNet/demo/korean_highway_noon/inference3/imgs"
device="cuda:0"
threshold=0.3

camera_mode=False
# below options are only valid when camera_mode sets to True
webcam_num=0
save_and_compose=False # whether to save the frames from the camera stream and compose the video from the saved frames
stream_save_path='/work/CLRerNet/demo/stream' # camera stream path to save the frame
stream_video_path='/work/CLRerNet/demo/stream/inference/video'


video_mode=True
# below options are only valid when video_mode sets to True
decompose=False # whether to do the decompose process
inference=True # whether to do the inference
compose=True # whether to do the compose process
decompose_video_path="/work/CLRerNet/demo/korean_highway_noon.mp4"
decompose_out_path="/work/CLRerNet/demo/korean_highway_noon/decomposed"
video_out_path="/work/CLRerNet/demo/korean_highway_noon/inference3/videos"
delete_imgs=True

python3 inference_tools.py --model_path $model_path --config_path $config_path \
    --img_path $img_path --out_dir $out_dir --device $device --threshold $threshold \
    --video_mode $video_mode --decompose $decompose --inference $inference --compose $compose \
    --decompose_video_path $decompose_video_path --decompose_out_path $decompose_out_path \
    --video_out_path $video_out_path --delete_imgs $delete_imgs --camera_mode $camera_mode --webcam_num $webcam_num \
    --save_and_compose $save_and_compose --stream_save_path $stream_save_path --stream_video_path $stream_video_path
