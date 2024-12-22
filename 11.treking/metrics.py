import numpy as np

def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    xmin1, xmin2 = bbox1[0], bbox2[0]
    ymin1, ymin2 = bbox1[1], bbox2[1]
    xmax1, xmax2 = bbox1[2], bbox2[2]
    ymax1, ymax2 = bbox1[3], bbox2[3]
    
    A_and_B = max(min(xmax1, xmax2) - max(xmin1, xmin2), 0) * max(min(ymax1, ymax2) - max(ymin1, ymin2), 0)
    A_or_B = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2)

    return A_and_B / (A_or_B - A_and_B)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = {idx: [xmin, ymin, xmax, ymax] for idx, xmin, ymin, xmax, ymax in frame_obj}
        dict_hyp = {idx: [xmin, ymin, xmax, ymax] for idx, xmin, ymin, xmax, ymax in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if obj_id in dict_obj and hyp_id in dict_hyp:
                iou_cur = iou_score(dict_obj[obj_id], dict_hyp[hyp_id])
                if iou_cur > threshold:
                    dist_sum += iou_cur
                    match_count += 1
                    dict_obj.pop(obj_id)
                    dict_hyp.pop(hyp_id)
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise_IOU = []
        for obj_id, obj in dict_obj.items():
            for hyp_id, hyp in dict_hyp.items():
                iou_cur = iou_score(obj, hyp)
                if iou_cur > threshold:
                    pairwise_IOU.append((iou_cur, obj_id, hyp_id))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairwise_IOU = sorted(pairwise_IOU, key=lambda x: x[0], reverse=True)
        for iou, obj_id, hyp_id in pairwise_IOU:
            if obj_id in dict_obj and hyp_id in dict_hyp:
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    dict_obj.pop(obj_id)
                    dict_hyp.pop(hyp_id)
                    # Step 5: Update matches with current matched IDs
                    matches[obj_id] = hyp_id

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.4):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    # print(len(obj))
    # print(obj)
    objects_all_idx = []
    for frame_obj in obj:
        for cur_obj in frame_obj:
            objects_all_idx.append(cur_obj[0])
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = {idx: [xmin, ymin, xmax, ymax] for idx, xmin, ymin, xmax, ymax in frame_obj}
        dict_hyp = {idx: [xmin, ymin, xmax, ymax] for idx, xmin, ymin, xmax, ymax in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if obj_id in dict_obj and hyp_id in dict_hyp:
                iou_cur = iou_score(dict_obj[obj_id], dict_hyp[hyp_id])
                if iou_cur > threshold:
                    dist_sum += iou_cur
                    match_count += 1
                    dict_obj.pop(obj_id)
                    dict_hyp.pop(hyp_id)
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise_IOU = []
        for obj_id, obj in dict_obj.items():
            for hyp_id, hyp in dict_hyp.items():
                iou_cur = iou_score(obj, hyp)
                if iou_cur > threshold:
                    pairwise_IOU.append((iou_cur, obj_id, hyp_id))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairwise_IOU = sorted(pairwise_IOU, key=lambda x: x[0], reverse=True)
        cur_step_matches = {}
        for iou, obj_id, hyp_id in pairwise_IOU:
            if obj_id in dict_obj and hyp_id in dict_hyp:
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    dict_obj.pop(obj_id)
                    dict_hyp.pop(hyp_id)
                    cur_step_matches[obj_id] = hyp_id
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        for obj_id, hyp_id in cur_step_matches.items():
            if obj_id in matches and matches[obj_id] != hyp_id:
                mismatch_error += 1
            # Step 6: Update matches with current matched IDs
            matches[obj_id] = hyp_id
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(dict_hyp.keys())
        # All remaining objects are considered misses
        missed_count += len(dict_obj.keys())
        # print(missed_count, " ", false_positive, " ", mismatch_error, " ", len(objects_all_idx))

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1. - (missed_count + false_positive + mismatch_error) / len(objects_all_idx)

    return MOTP, MOTA
