import os
import glob
from io import BytesIO
import base64
from PIL import Image
import json


def resize_images_and_labels(labelme_json_paths, output_dir, size):
    os.makedirs(output_dir, exist_ok=True)

    for json_file in labelme_json_paths:
        # Open json file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Load base64 image
        im = Image.open(BytesIO(base64.b64decode(data['imageData'])))

        # Resize image
        im_resized = im.resize(size, Image.ANTIALIAS)

        # Change imageHeight and imageWidth in json
        data['imageWidth'] = size[0]
        data['imageHeight'] = size[1]

        # Change imageData
        buffered = BytesIO()
        im_resized.save(buffered, format="JPEG")
        data['imageData'] = base64.b64encode(buffered.getvalue()).decode()

        # Change datapoints
        width_ratio = im_resized.size[0] / im.size[0]
        height_ratio = im_resized.size[1] / im.size[1]
        for annotation in data['shapes']:
            resized_points = []
            for point in annotation['points']:
                resized_points.append([point[0] * width_ratio, point[1] * height_ratio])
            annotation['points'] = resized_points

        # Save image
        im_resized.save(os.path.join(output_dir, data['imagePath']))

        # Save json file
        with open(os.path.join(output_dir, os.path.basename(json_file)), 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Resize size of images with labelme labels')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory to labelme images and annotation json files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to directory where new images and labels will be saved')
    parser.add_argument('--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()

    labelme_json = glob.glob(os.path.join(args.input_dir, "*.json"))
    resize_images_and_labels(labelme_json, args.output_dir, args.size)
