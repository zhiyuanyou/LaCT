# Resolution 536 x 960, Num input views 32:
python inference.py \
--load weight/scene_res536x960.pt \
--config config/lact_l24_d768_ttt4x.yaml \
--image_size 536 960 \
--scene_inference \
--num_input_views 48 \
--data_path data_example/dl3dv_sample_data_path.json \
--best_path data_example/dl3dv_sample_best_path.json \
--output_dir output_max10/ \
--max_yaw_deg 10 \
--max_pitch_deg 10 \
--num_yaw 5 \
--num_pitch 5 \
--max_rectify_deg 10
