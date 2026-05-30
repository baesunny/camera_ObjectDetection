"""IoU utilities for stable object tracking across frames."""


def intersection_over_union(box_a, box_b):
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    intersection_area = max(x_max - x_min, 0) * max(y_max - y_min, 0)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    total_area = box_a_area + box_b_area - intersection_area
    return intersection_area / total_area


def distance(box_a, box_b):
    center_a_x = (box_a[0] + box_a[2]) / 2
    center_a_y = (box_a[1] + box_a[3]) / 2
    center_b_x = (box_b[0] + box_b[2]) / 2
    center_b_y = (box_b[1] + box_b[3]) / 2
    return (center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2


def make_dict(a):
    result = {}
    for i in range(1, a + 1):
        for j in range(i + 1):
            if j == 0:
                result[(i, j)] = [[]]
            elif j == 1:
                result[(i, j)] = [[k] for k in range(i)]
            else:
                cnt = 0
                nodes = [[k] for k in range(i)]
                candidates = list(range(i))
                while cnt < i * j:
                    node = nodes.pop(0)
                    if len(node) == j:
                        cnt += 1
                        nodes.append(node)
                    else:
                        for num in candidates:
                            if num not in node:
                                new_node = node.copy()
                                new_node.append(num)
                                nodes.append(new_node)
                result[(i, j)] = sorted(nodes)
    return result


def iou_multiple(boxes_a, boxes_b, view_threshold=3, t=2):
    possible_idx_dict = make_dict(view_threshold)

    iou_total_list = []
    len_a = len(boxes_a)
    len_b = len(boxes_b)
    iou_store = []

    if len_b > len_a:
        len_a, len_b = len_b, len_a
        boxes_a, boxes_b = boxes_b, boxes_a

    t_scaled_pct_sum = sum(sublist[-1] ** (1 / t) for sublist in boxes_a) + sum(
        sublist[-1] ** (1 / t) for sublist in boxes_b
    )

    for indices in possible_idx_dict[(len_a, len_b)]:
        iou_list = []
        if len_b == 0:
            iou_list.append(0)
        else:
            for j in range(len_b):
                if boxes_b[j][-2] == boxes_a[indices[j]][-2]:
                    iou_list.append(
                        intersection_over_union(boxes_b[j][:-1], boxes_a[indices[j]][:-1])
                    )
                else:
                    iou_list.append(0)
        iou_total_list.append(sum(iou_list) / len(iou_list))
        iou_store.append(iou_list)

    max_index = iou_total_list.index(max(iou_total_list))
    cal_index_order = possible_idx_dict[(len_a, len_b)][max_index]

    iou_a, iou_b = [], []
    for i in range(len_a):
        if i not in cal_index_order:
            iou_a.append([0, boxes_a[i][-1]])
        else:
            iou_a.append([iou_store[max_index][cal_index_order.index(i)], boxes_a[i][-1]])
    for i in range(len_b):
        iou_b.append([iou_store[max_index][i], boxes_b[i][-1]])

    weighted_iou = 0
    for iou, pct in iou_a:
        weighted_iou += (pct ** (1 / t)) / t_scaled_pct_sum * iou
    for iou, pct in iou_b:
        weighted_iou += (pct ** (1 / t)) / t_scaled_pct_sum * iou
    return weighted_iou
