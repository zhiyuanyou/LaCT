import json
import os


if __name__ == "__main__":
    threshold_score = 0.2
    threshold_idx = 3
    num_best = 3
    root_dir = "data_example/dl3dv_processed"
    data_path = "data_example/dl3dv_sample_data_path.json"
    save_path = "data_example/dl3dv_sample_best_path.json"

    with open(data_path, "r") as f:
        data_list = json.load(f)
    
    best_paths = []
    for data_path in data_list:
        scene_id = os.path.basename(os.path.dirname(data_path))
        best_path = os.path.join(root_dir, scene_id, "best_views.json")
        best_paths.append(best_path.replace(root_dir, "dl3dv_processed"))

        if os.path.exists(best_path):
            print(f"Skipping {best_path} because it already exists")
            continue
        
        eval_path = os.path.join(root_dir, scene_id, "eval_score.json")
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        image_score_pairs = []
        for item in eval_data:
            if item["extracted_answer"] is not None:
                # frame_00001.jpg -> 1
                idx = int(item["image"].split("_")[-1].split(".")[0])
                image_score_pairs.append((item["extracted_answer"], item["image"], idx))

        # 根据score进行排序
        image_score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # 分离image_list和score_list
        score_list = [pair[0] for pair in image_score_pairs]
        image_list = [pair[1] for pair in image_score_pairs]
        idx_list = [pair[2] for pair in image_score_pairs]

        best_views = []
        best_views_idx = []
        best_score = max(score_list)
        best_score_min = best_score - threshold_score
        for image, score, idx in zip(image_list, score_list, idx_list):
            if score >= best_score_min:
                # 检查是否与已有的idx间隔在threshold_idx之内
                is_too_close = False
                for existing_idx in best_views_idx:
                    if abs(idx - existing_idx) <= threshold_idx:
                        is_too_close = True
                        break
                
                # 只有当不与现有idx冲突时才添加
                if not is_too_close:
                    best_views.append(os.path.join("images_undistort", image))
                    best_views_idx.append(idx)

        best_views = best_views[:num_best]
        with open(best_path, "w") as fw:
            fw.write(json.dumps(best_views, indent=2))

    with open(save_path, "w") as fw:
        fw.write(json.dumps(best_paths, indent=2))
