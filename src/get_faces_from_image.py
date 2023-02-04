import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import glob
import argparse
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--faces", default=20,
                help="Number of faces that camera will get")
ap.add_argument("--output", default="../datasets/train",
                help="Path to faces output")
ap.add_argument("--images-dir", default="../datasets/images",
    help='Path to output image')
args = vars(ap.parse_args())

# Detector = mtcnn_detector
detector = MTCNN()

image_dir = args['images_dir']
persons = glob.glob(image_dir+ '/**')
for person in persons:
    person_name = os.path.basename(person)
    img_count = 0;
    imgs = glob.glob(person+"/*.jpg")
    for img in imgs:
        # Get all faces on current image
        img = cv2.imread(img)
        bboxes = detector.detect_faces(img)
        if len(bboxes) != 0:
            # Get only the biggest face
            max_area = 0
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                keypoints = bboxe["keypoints"]
                area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                if area > max_area:
                    max_bbox = bbox
                    landmarks = keypoints
                    max_area = area

            max_bbox = max_bbox[0:4]

            # convert to face_preprocess.preprocess input
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                    landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2,5)).T
            nimg = face_preprocess.preprocess(img, max_bbox, landmarks, image_size='112,112')
            output_path = os.path.join(args['output'], person_name)
            if not(os.path.exists(output_path)):
                os.makedirs(output_path)
            # if not(os.path.exists(args["output"])):
            #     os.makedirs(args["output"])
            cv2.imwrite(os.path.join(output_path, "{}.jpg".format(img_count+1)), nimg)
            cv2.rectangle(img, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
            # print("[INFO] {} faces detected".format(img_count+1))
            img_count += 1


# cv2.imshow("Face detection", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()