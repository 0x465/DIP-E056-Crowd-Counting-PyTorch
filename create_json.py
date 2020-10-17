import os
import json
import glob

if __name__ == '__main__':
    # path to folder that contains images
    root = '...input/shanghaitech_with_people_density_map/ShanghaiTech'
    json_path = '...input/json'

    paths = []
    paths.append(os.path.join(root, 'part_A/train_data', 'images'))
    paths.append(os.path.join(root, 'part_A/test_data', 'images'))
    paths.append(os.path.join(root, 'part_B/train_data', 'images'))
    paths.append(os.path.join(root, 'part_B/test_data', 'images'))
    #[print(p) for p in paths]

    for i, path in enumerate(paths):
        if i == 0:
            output_json = os.path.join(json_path, 'part_A_train.json')
        elif i == 1:
            output_json = os.path.join(json_path, 'part_A_test.json')
        elif i == 2:
            output_json = os.path.join(json_path, 'part_B_train.json')
        elif i == 3:
            output_json = os.path.join(json_path, 'part_B_test.json')

        img_list = []

        for img_path in glob.glob(join(path,'*.jpg')):
            img_list.append(img_path)

        with open(output_json,'w') as f:
            json.dump(img_list,f)