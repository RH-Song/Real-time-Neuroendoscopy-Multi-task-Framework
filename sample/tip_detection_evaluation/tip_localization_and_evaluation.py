import tip_localization as tl
import txt_result_operator as tp
import tip_evaluation as te
from os import rmdir
from os.path import exists, join
import shutil

if __name__ == '__main__':
    # configs
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/deeplabv3"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/deeplabv3_svm_93"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/fcn8s"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/fcn8s_svm_95"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/ground_truth"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp"
    origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/psp_svm_95"
    #origin_path = "/home/raphael/Desktop/data/surgery/surgery_seg_results/maskrcnn_svm"
    origin_image_dir = "/home/raphael/Desktop/data/surgery/surgery_seg_dataset/img"
    annotation_dir = '/home/raphael/Desktop/data/surgery/Tip/refactted_annotations'

    REMOVE_RESULTS = True
    if REMOVE_RESULTS:
        if exists(join(origin_path, 'results')):
            shutil.rmtree(join(origin_path, 'results'))

    tl.tip_localization(origin_path, origin_image_dir)
    tp.format_raw_results(origin_path)
    tp.split_result(origin_path)
    te.tip_evaluation(origin_path, annotation_dir)
