# function to return IoU
def intersection_over_union(boxA,boxB):
    # xmin, ymin, xmax, ymax of intersection box
    x_min = max(boxA[0],boxB[0])
    y_min = max(boxA[1],boxB[1])
    x_max = min(boxA[2],boxB[2])
    y_max = min(boxA[3],boxB[3])

    # Intersection 영역
    intersection_area = max(x_max-x_min,0) * max(y_max-y_min,0)
    # boxA, boxB 영역
    boxA_area = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    # Total 영역
    total_area = boxA_area + boxB_area - intersection_area
    IoU = intersection_area/total_area
    return IoU



# function to return distance btw center coordinates
def distance(boxA, boxB):
    center_A_x = (boxA[0]+boxA[2])/2
    center_A_y = (boxA[1]+boxA[3])/2
    center_B_x = (boxB[0]+boxB[2])/2
    center_B_y = (boxB[1]+boxB[3])/2

    # distance : 유클리드 거리의 제곱
    distance = (center_A_x-center_B_x)^2 + (center_A_y-center_B_y)^2
    return distance