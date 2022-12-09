'''
Evalutate tip detection algorithm.
'''
import txt_result_operator
from os import listdir
from os.path import join
import json

def match_tip(tip_truth, tip_points, threshold):
    match_result = False
    nearest_distance = 2160
    nearest_tip = []
    for tip in tip_points:
        # calculate Euclidean distance
        distance = ((tip[0] - tip_truth[0]) ** 2 + (tip[1] - tip_truth[1]) ** 2) ** 0.5
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_tip = tip
    if nearest_distance < threshold:
        match_result = True
        tip_points.remove(nearest_tip)
        #print('removed: {}'.format(nearest_tip))
    return match_result, tip_points

def tip_evaluation(origin_dir, annotation_dir):
    print('start evaluation')
    origin_dir = origin_dir + '/results'
    results_dir = origin_dir + "/singles"
    precision_file = 'precison.txt'
    recall_file = 'recall.txt'
    AF_file = 'af_pre_results.txt'
    AF_recall_file = 'af_recall_results.txt'
    BDEH_file = 'bdeh_pre_results.txt'
    BDEH_recall_file = 'bdeh_recall_results.txt'
    C_file = 'c_pre_results.txt'
    C_recall_file = 'c_recall_results.txt'
    AF_txt = open(join(origin_dir, AF_file), 'a')
    AF_recall_txt = open(join(origin_dir, AF_recall_file), 'a')
    BDEH_txt = open(join(origin_dir, BDEH_file), 'a')
    BDEH_recall_txt = open(join(origin_dir, BDEH_recall_file), 'a')
    C_txt = open(join(origin_dir, C_file), 'a')
    C_recall_txt = open(join(origin_dir, C_recall_file), 'a')
    pre_txt = open(join(origin_dir, precision_file), 'a')
    recall_txt = open(join(origin_dir, recall_file), 'a')

    METHOD = 'FCN\n'
    pre_txt.write(METHOD)
    recall_txt.write(METHOD)
    AF_txt.write(METHOD)
    AF_recall_txt.write(METHOD)
    BDEH_txt.write(METHOD)
    BDEH_recall_txt.write(METHOD)
    C_txt.write(METHOD)
    C_recall_txt.write(METHOD)

    for threshold in range(10, 180, 10):
        print(threshold)

        total_Pre = 0
        AF_pre = 0
        BDEH_pre = 0
        c_pre = 0
        total_recall = 0
        AF_recall = 0
        BDEH_recall = 0
        c_recall = 0
        AF_num = 0
        BDEH_num = 0
        C_num = 0

        results_objs = listdir(results_dir)
        annotation_objs = listdir(annotation_dir)
        num_of_annotations = len(annotation_objs)
        num_of_results = len(results_objs)

        traverse_objs = results_objs
        num_of_obj = len(traverse_objs)
        #print(num_of_obj)

        for traverse_obj in traverse_objs:
            name, type = traverse_obj.split('.')
            #print(name)
            result_obj = name + '.txt'
            # Get tip points from txt
            result = open(join(results_dir, result_obj), 'r')
            tip_points = txt_result_operator.get_tip(result)
            #print('prediction')
            #print(tip_points)
            result.close()

            # Get tip points from json annotation
            tip_truths = []
            annotation_obj = name + '.json'
            with open(join(annotation_dir, annotation_obj)) as src_f:
                annotation_dict = json.load(src_f)
                shapes = annotation_dict["shapes"]
                for shape in shapes:
                    if shape['label'] == 'tip':
                        tip_truth = shape['points']
                        tip_truths.append(tip_truth[0])

            #print('truth')
            #print(tip_truths)
            num_of_true = 0
            num_of_prediction = len(tip_points)
            num_of_gt = len(tip_truths)

            for tip_truth in tip_truths:
                match_result, tip_points = match_tip(tip_truth, tip_points, threshold)
                #print('new_tip: {}'.format(tip_points))
                if match_result:
                    num_of_true += 1

            recall = num_of_true / num_of_gt
            if num_of_prediction == 0:
                precision = 0
            else:
                precision = num_of_true / num_of_prediction
            total_Pre += precision
            total_recall += recall

            index_of_file = int(name)
            if index_of_file >= 861 and index_of_file <= 1214:
                AF_pre += precision
                AF_recall += recall
                AF_num += 1
            elif index_of_file >= 2631 and index_of_file <= 3124:
                BDEH_pre += precision
                BDEH_recall += recall
                BDEH_num += 1
            elif index_of_file >= 802 and index_of_file <= 860:
                c_pre += precision
                c_recall += recall
                C_num += 1
            elif index_of_file >= 2103 and index_of_file <= 2630:
                BDEH_pre += precision
                BDEH_recall += recall
                BDEH_num += 1
            elif index_of_file >= 1215 and index_of_file <= 1694:
                BDEH_pre += precision
                BDEH_recall += recall
                BDEH_num += 1
            elif index_of_file >= 1695 and index_of_file <= 2102:
                AF_pre += precision
                AF_recall += recall
                AF_num += 1
            elif index_of_file >= 1 and index_of_file <= 801:
                BDEH_pre += precision
                BDEH_recall += recall
                BDEH_num += 1

            #print("precision: {}, recall: {}".format(precision, recall))

        #print('num of obj: {}'.format(num_of_obj))
        #print('total pre: {}'.format(total_Pre))
        #print('total recall: {}'.format(total_recall))
        mPre = total_Pre / num_of_obj
        mRecal = total_recall / num_of_obj
        mAF = AF_pre / AF_num
        mAF_recall = AF_recall / AF_num
        mBDEH = BDEH_pre / BDEH_num
        mBDEH_recall = BDEH_recall / BDEH_num
        mC = c_pre / C_num
        mC_recall = c_recall / C_num
        print("=========================")
        print("average precision: {}, average recall: {}".format(mPre, mRecal))
        pre_txt.write(str(mPre)+'\n')
        recall_txt.write(str(mRecal)+'\n')
        AF_txt.write(str(mAF)+'\n')
        AF_recall_txt.write(str(mAF_recall)+'\n')
        BDEH_txt.write(str(mBDEH)+'\n')
        BDEH_recall_txt.write(str(mBDEH_recall)+'\n')
        C_txt.write(str(mC)+'\n')
        C_recall_txt.write(str(mC_recall)+'\n')
    pre_txt.close()
    recall_txt.close()
    AF_txt.close()
    AF_recall_txt.close()
    BDEH_txt.close()
    BDEH_recall_txt.close()
    C_txt.close()
    C_recall_txt.close()


if __name__ == '__main__':
    #config
    #origin_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp"
    origin_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp_svm"
    #origin_dir = '/home/raphael/Desktop/data/surgery/surgery_seg_results/results_11-1'
    #origin_dir = '/home/raphael/Desktop/data/surgery/surgery_seg_results/ground_truth'
    #origin_dir = '/home/raphael/Desktop/data/surgery/surgery_seg_results/deeplabv3'
    #origin_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_results/fcn8s"
    annotation_dir = '/home/raphael/Desktop/data/surgery/Tip/refactted_annotations'

    tip_evaluation(origin_dir, annotation_dir)
