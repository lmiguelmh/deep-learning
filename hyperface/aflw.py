import sqlite3
import os.path
import numpy as np
import math
import cv2
import dlib

'''
adapted and based mainly from: takiyu, chainer
'''

N_LANDMARK = 21
IMG_SIZE = (227, 227)


def exec_sqlite_query(cursor, select_str, from_str=None, where_str=None):
    query_str = 'SELECT {}'.format(select_str)
    query_str += ' FROM {}'.format(from_str)
    if where_str:
        query_str += ' WHERE {}'.format(where_str)
    return [row for row in cursor.execute(query_str)]


def load_raw_aflw(sqlite_path, image_dir):
    """
    Load raw AFLW dataset from sqlite file
    Return:
        [dict('face_id', 'img_path', 'rect', 'landmark', 'landmark_visib',
              'pose', 'gender')]
    """
    print('Load raw AFLW dataset from "{}"'.format(sqlite_path))

    # Temporary dataset variables
    dataset_dict = dict()

    # Open sqlite file
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Basic property
    select_str = "faces.face_id, imgs.filepath, " \
                 "rect.x, rect.y, rect.w, rect.h, " \
                 "pose.roll, pose.pitch, pose.yaw, metadata.sex"
    from_str = "faces, faceimages imgs, facerect rect, facepose pose, " \
               "facemetadata metadata"
    where_str = "faces.file_id = imgs.file_id and " \
                "faces.face_id = rect.face_id and " \
                "faces.face_id = pose.face_id and " \
                "faces.face_id = metadata.face_id"
    query_res = exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    for face_id, path, rectx, recty, rectw, recth, roll, pitch, yaw, gender in query_res:
        # Data creation or conversion
        img_path = os.path.join(image_dir, path) if image_dir else path
        landmark = np.zeros((N_LANDMARK, 2), dtype=np.float32)
        landmark_visib = np.zeros(N_LANDMARK, dtype=np.float32)
        pose = np.array([roll, pitch, yaw], dtype=np.float32)
        gender = np.array([1, 0] if gender == 'm' else [0, 1], dtype=np.int32)
        others_landmark_pts = list()
        # Register
        data = {'face_id': face_id,
                'img_path': img_path,
                'face_rect': (rectx, recty, rectw, recth),
                'landmark': landmark,
                'landmark_visib': landmark_visib,
                'pose': pose,
                'gender': gender,
                'others_landmark_pts': others_landmark_pts}
        dataset_dict[face_id] = data

    # Landmark property
    # (Visibility is expressed by lack of the coordinate's row.)
    select_str = "faces.face_id, coords.feature_id, " \
                 "coords.x, coords.y"
    from_str = "faces, featurecoords coords"
    where_str = "faces.face_id = coords.face_id"
    query_res = exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    invalid_face_ids = list()
    for face_id, feature_id, x, y in query_res:
        assert (1 <= feature_id <= N_LANDMARK)
        if face_id in dataset_dict:
            idx = feature_id - 1
            dataset_dict[face_id]['landmark'][idx][0] = x
            dataset_dict[face_id]['landmark'][idx][1] = y
            dataset_dict[face_id]['landmark_visib'][idx] = 1
        elif face_id not in invalid_face_ids:
            print('Invalid face id ({}) in AFLW'.format(face_id))
            invalid_face_ids.append(face_id)

    # Landmarks of other faces, file_id could have more than one face_id, thats why !=
    select_str = "a.face_id, coords.x, coords.y"
    from_str = "faces a, faces b, featurecoords coords"
    where_str = "a.face_id != b.face_id and a.file_id = b.file_id and " \
                "b.face_id = coords.face_id"
    query_res = exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    for face_id, others_x, others_y in query_res:
        if face_id in dataset_dict:
            other_coord = [others_x, others_y]
            dataset_dict[face_id]['others_landmark_pts'].append(other_coord)
        else:
            assert (face_id in invalid_face_ids)
    # Convert list to np.ndarray
    for data in dataset_dict.values():
        pts = np.array(data['others_landmark_pts'], dtype=np.float32)
        data['others_landmark_pts'] = pts

    # Exit sqlite
    cursor.close()

    # Return dataset_dict's value (list)
    return list(dataset_dict.values())


def rect_contain(rect, pt):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def extract_valid_rects(rects, img, others_landmark_pts):
    """
    Extract rectangles which do not contain other landmarks
    """
    # Extraction
    dst = list()
    for rect in rects:
        # Check if others landmarks are contained
        for others_pt in others_landmark_pts:
            if rect_contain(rect, others_pt):
                break
        else:
            dst.append(rect)

    # avoid no rectangle
    if len(dst) == 0:
        dst.append((0, 0, img.shape[1], img.shape[0]))

    return dst


def rect_overlap_rate(a, b):
    area_and = rect_area(rect_and(a, b))
    area_or = rect_area(rect_or(a, b))
    if area_or == 0:
        return 0
    else:
        return math.sqrt(float(area_and) / float(area_or))


def rect_or(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def rect_and(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return (0, 0, 0, 0)
    return (x, y, w, h)


def rect_area(a):
    return a[2] * a[3]


def setup_raw_aflw(raw_dataset,
                   log_interval=10,
                   max_iterations=101,
                   overlap_positive_face_rate=0.50,
                   overlap_negative_face_rate=0.25,
                   apply_nms=True,
                   ss_max_img_size=(500, 500),
                   ss_kvals=(50, 200, 2),
                   ss_min_size=2200):
    # Calculate selective search rectangles (This takes many minutes)
    print('Calculate selective search rectangles for AFLW')
    positive_faces = []
    negative_faces = []
    positive_not_found_count = 0
    for i, entry in enumerate(raw_dataset):
        if i >= max_iterations:
            break

        if i % log_interval == 0:
            print(' {}/{}'.format(i, len(raw_dataset)))

        # Load image
        img_path = entry['img_path']
        # print('loading image', img_path)
        img = cv2.imread(img_path)

        if img is None or img.size == 0:
            # Empty elements
            print('failed to load image {}'.format(img_path))
            raw_dataset[i]['ssrects'] = list()
            raw_dataset[i]['ssrect_overlaps'] = list()
        else:
            # Selective search
            ssrects = selective_search_dlib(img, ss_max_img_size, ss_kvals, ss_min_size)
            # Extract rectangles which do not contain other landmarks
            ssrects = extract_valid_rects(ssrects, img, entry['others_landmark_pts'])
            positive_ssrects = [ssrect for ssrect in ssrects if
                                rect_overlap_rate(ssrect, entry['face_rect']) > overlap_positive_face_rate]
            negative_ssrects = [ssrect for ssrect in ssrects if
                                rect_overlap_rate(ssrect, entry['face_rect']) < overlap_negative_face_rate]
            if apply_nms:
                # print('before nms pos/neg: %d/%d' % (len(positive_ssrects), len(negative_ssrects)))
                positive_ssrects = non_max_suppression_fast(np.array(positive_ssrects), 0.5)
                negative_ssrects = non_max_suppression_fast(np.array(negative_ssrects), 0.5)
                # print('after nms pos/neg: %d/%d' % (len(positive_ssrects), len(negative_ssrects)))

            if len(positive_ssrects) == 0:
                positive_not_found_count = positive_not_found_count + 1

            for ssrect in positive_ssrects:
                positive_faces.append({'labelFnf': 1,
                                       'labelLandmarks': entry['landmark'].tolist(),
                                       'labelVisFac': entry['landmark_visib'].tolist(),
                                       'labelPose': entry['pose'].tolist(),
                                       'labelGender': entry['gender'].tolist(),
                                       'image': img_path,
                                       'bbox': ssrect.tolist()})

            for ssrect in negative_ssrects:
                negative_faces.append({'labelFnf': 0,
                                       'image': img_path,
                                       'bbox': ssrect.tolist()})

                # for ssrect in ssrects:
                #     overlap = rect_overlap_rate(ssrect, entry['face_rect'])
                #     if overlap > overlap_positive_face_rate:
                #         positive_faces.append({'labelFnf': 1,
                #                                'labelLandmarks': entry['landmark'].tolist(),
                #                                'labelVisFac': entry['landmark_visib'].tolist(),
                #                                'labelPose': entry['pose'].tolist(),
                #                                'labelGender': entry['gender'].tolist(),
                #                                'image': entry['img_path'],
                #                                'bbox': ssrect})
                #     elif overlap < overlap_negative_face_rate:
                #         negative_faces.append({'labelFnf': 0,
                #                                'image': entry['img_path'],
                #                                'bbox': ssrect})

    print('Found %d positive samples and %d negative samples.' % (len(positive_faces), len(negative_faces)))
    print('%d images have 0 positives.' % (positive_not_found_count))
    return positive_faces, negative_faces


def scale_down_image(img, max_img_size):
    org_h, org_w = img.shape[0:2]
    h, w = img.shape[0:2]
    if max_img_size[0] < w:
        h *= float(max_img_size[0]) / float(w)
        w = max_img_size[0]
    if max_img_size[1] < h:
        w *= float(max_img_size[1]) / float(h)
        h = max_img_size[1]
    # Apply resizing
    if h == org_h and w == org_w:
        resize_scale = 1
    else:
        resize_scale = float(org_h) / float(h)  # equal to `org_w / w`
        img = cv2.resize(img, (int(w), int(h)))
    return img, resize_scale


def selective_search_dlib(img,
                          max_img_size=(500, 500),
                          kvals=(50, 200, 2),
                          min_size=2200,
                          check=True):
    org_h, org_w = img.shape[0:2]

    # Resize the image for speed up
    img, resize_scale = scale_down_image(img, max_img_size)

    # dlib' selective search
    # http://dlib.net/dlib/image_transforms/segment_image_abstract.h.html#find_candidate_object_locations
    drects = []
    dlib.find_candidate_object_locations(img,
                                         drects,
                                         kvals=kvals,
                                         min_size=min_size)
    rects = [(int(drect.left() * resize_scale),
              int(drect.top() * resize_scale),
              int(drect.width() * resize_scale),
              int(drect.height() * resize_scale)) for drect in drects]

    # Check the validness of the rectangles
    if check:
        if len(rects) == 0:
            print('No selective search rectangle (Please tune the parameters)')
        for rect in rects:
            x, y = rect[0], rect[1]
            w, h = rect[2], rect[3]
            x2, y2 = x + w, y + h
            if x < 0 or y < 0 or org_w < x2 or org_h < y2 or w <= 0 or h <= 0:
                print('Invalid selective search rectangle, rect:{}, image:{}'.format(rect, (org_h, org_w)))
    return rects


def bb_intersection_over_union(boxA, boxB):
    """
    Intersection over Union
    http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param boxA:
    :param boxB:
    :return:
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def non_max_suppression_fast(boxes, overlapThresh):
    """
    Non-Maxima Suppression implementation from Malisiewicz et al.:
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    :param boxes:
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
