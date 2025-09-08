import json
import os


if __name__ == "__main__":
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
            continue
        
        eval_path = os.path.join(root_dir, scene_id, "eval_score.json")
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        image_list = []
        score_list = []
        for item in eval_data:
            if item["extracted_answer"] is not None:
                image_list.append(item["image"])
                score_list.append(item["extracted_answer"])

        best_views = []
        best_score = max(score_list)
        for image, score in zip(image_list, score_list):
            if score == best_score:
                best_views.append(os.path.join("images_undistort", image))
        best_views = best_views[:num_best]
        with open(best_path, "w") as fw:
            fw.write(json.dumps(best_views, indent=2))

    with open(save_path, "w") as fw:
        fw.write(json.dumps(best_paths, indent=2))
