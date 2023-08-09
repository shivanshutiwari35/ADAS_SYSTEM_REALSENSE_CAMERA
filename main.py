import cv2
from realsense_camera import*
from mask_rcnn import*

#load realsense camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:

    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    mrcnn.draw_object_info(bgr_frame, depth_frame)

    mean_value = (np.mean(depth_frame))/100
    print("Mean value in cms:", mean_value)

    if mean_value < 100:
        print("Stop")

    else:
        print("Go")


    cv2.imshow("Depth_frame", depth_frame)
    cv2.imshow("BGR Frame", bgr_frame)

    key = cv2.waitKey(1)

    if key ==65:
        break

