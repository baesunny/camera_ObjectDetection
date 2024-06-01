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

def make_dict(a):
    dict = {}
    for i in range(1, a+1):
        for j in range(i+1):
            if j == 0:
                li = [[]]
                dict[(i,j)] = li
            elif j == 1:
                li = [[k] for k in range(i)]
                dict[(i,j)] = li
            else :
                cnt = 0
                li = [[k] for k in range(i)]
                can = [k for k in range(i)]
                while cnt < (i)*(j):
                    node = li.pop(0)
                    if len(node) == j:
                        cnt+=1
                        li.append(node)
                    else :
                        for num in can:
                            if num not in node:
                                new_node = node.copy()
                                new_node.append(num)
                                li.append(new_node)
                dict[(i,j)] = sorted(li)
    return dict

def iou_multiple(boxesA, boxesB, view_threshold=3, t=2):
    possible_idx_dict=make_dict(view_threshold)
    
    iou_total_list = []
    lenA = len(boxesA)
    lenB = len(boxesB)
    iou_store = []
    
    if lenB > lenA:
        lenA, lenB = lenB, lenA
        boxesA, boxesB = boxesB, boxesA
    
    t_scaled_pct_sum = sum((sublist[-1])**(1/t) for sublist in boxesA) + sum((sublist[-1])**(1/t) for sublist in boxesB)
    
    for i in possible_idx_dict[(lenA, lenB)]:
        iou_list=[]
        if lenB == 0:
            iou_list.append(0)
        else:
            for j in range(lenB):
                if boxesB[j][-2] == boxesA[i[j]][-2]:
                    iou_list.append(intersection_over_union(boxesB[j][:-1], boxesA[i[j]][:-1]))
                else:
                    iou_list.append(0)
        iou_total_list.append(sum(iou_list)/len(iou_list))
        iou_store.append(iou_list)
        
    max_index = iou_total_list.index(max(iou_total_list))
    cal_index_oreder = possible_idx_dict[(lenA, lenB)][max_index]

    iouA, iouB = [], []
    for i in range(lenA):
        if i not in cal_index_oreder:
            iouA.append([0, boxesA[i][-1]])
        else:
            iouA.append([iou_store[max_index][cal_index_oreder.index(i)], boxesA[i][-1]])
    for i in range(lenB):
        iouB.append([iou_store[max_index][i], boxesB[i][-1]])
    
    weighted_iou = 0
    for iou, pct in iouA:
        weighted_iou += (pct)**(1/t)/t_scaled_pct_sum*iou
    for iou, pct in iouB:
        weighted_iou += (pct)**(1/t)/t_scaled_pct_sum*iou
    # return max(iou_total_list), weighted_iou
    return weighted_iou

# boxesA=[[0,0,5,5,4,50],[0,0,3,3,1,18], [0,0,1,3,2,6]]
# boxesB=[[0,0,4,5,4,40], [0,0,2,3,1,12]]
# print(iou_multiple(boxesA, boxesB))
# print(iou_multiple(boxesA, boxesB, t=1))