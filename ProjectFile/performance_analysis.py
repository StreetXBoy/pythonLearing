# -*- coding: utf-8 -*-
'''
Calculate the overall accuracy and general accuracy
Categorize confusion error type
Generate a performance report that includes precision error
Generate lists of SKUs that need image retreival and remove others

Author: Yang Le
Time: Dec 11, 2019
Version 3.0

Example: python performance_analysis.py -if /path/to/debug_results.csv -op /datadrive/results/retail/e2e_retail_$JOBNAME/
                                        -hs /path/to/head_sku.csv -hsb /path/to/head_subbrand.csv -meta /path/to/metadata.csv
                                        -ta 0.92 -tsn 200 -esn 0 -tc final_target_sku.csv
                                        -sp ~/caffe/examples/retail/china_drink_sku_pool_eval -pr 75.0
'''
import pandas as pd
import codecs
import os
import argparse
import math
import itertools

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(
    description='Read dimension_result to generate analysis')
parser.add_argument(
    '-if', '--input_file', type=str, required=True,
    help='Local location for dimension result')
parser.add_argument(
    '-op', '--output_path', type=str, required=True,
    help='Local location for output files')
parser.add_argument(
    '-hs', '--head_sku', type=str, required=True,
    help='Location of head SKU file')
parser.add_argument(
    '-hsb', '--head_subbrand', type=str, required=True,
    help='Location of head subbrand file')
parser.add_argument(
    '-meta', '--metadata', type=str, required=True,
    help='Location of metadata file')
parser.add_argument(
    '-ta', '--target_accuracy', type=float, default=0.0,
    help='Target accuracy after QA')
parser.add_argument(
    '-tsn', '--train_sku_num', type=int, required=True,
    help = 'The threshold to consider train data sufficient')
parser.add_argument(
    '-esn', '--eval_sku_num', type=int, required=True,
    help = 'The threshold to consider eval data sufficient')
parser.add_argument(
    '-tc', '--train_cal', type=str, required=True,
    help = 'The training count for all SKUs')
parser.add_argument(
    '-pr', '--pr_thresh', type=float, default=85.0,
    help='The p/r threshold for onboard')
parser.add_argument(
    '-sp', '--sku_pool', type=str,
    help='The path to the SKU pool for evaluation')
parser.add_argument(
    '-stc', '--subbrand_train_count', type=str, default="",
    help='The path to the subbrand training count')
parser.add_argument(
    '-label', '--sku_labels', default='examples/retail/pms_sku_labels.csv',
    help='SKU labels file')
args = parser.parse_args()

df = pd.DataFrame()

subbrand_map = dict()
volume_map = dict()
unit_map = dict()
package_map = dict()
series_map = dict()
flavor_map = dict()
pool_map = dict()
sku_precision_error = dict()
sku_recall_error = dict()

def get_unit_perf(df, head_sku=[], head_subbrand=[]):
    # Unit level calculation
    predicted_unmatched = df[df['EvalResult']=='predict_unmatched']
    truth_unmatched = df[df['EvalResult']=='truth_unmatched']
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        predicted_unmatched_wrong = predicted_unmatched[predicted_unmatched['TopLabel'].isin(head_sku+head_subbrand)].shape[0]
        predicted_unmatched_others = predicted_unmatched[~predicted_unmatched['TopLabel'].isin(head_sku+head_subbrand)].shape[0]
        truth_unmatched_head = truth_unmatched[truth_unmatched['TrueLabel'].isin(head_sku + head_subbrand)].shape[0]
        truth_unmatched_others = truth_unmatched[~truth_unmatched['TrueLabel'].isin(head_sku + head_subbrand)].shape[0]
        return predicted_unmatched_wrong, truth_unmatched_head, predicted_unmatched_others, truth_unmatched_others

    return predicted_unmatched.shape[0], truth_unmatched.shape[0], 0, 0

def get_sku_perf(df, head_list=[]):
    # Get count for each confusion scenario
    correct = df[df['EvalResult']=='correct'].shape[0]
    all_wrong = df[df['EvalResult']=='wrong']
    if len(head_list) > 0:
        other = all_wrong[~all_wrong['TopLabel'].isin(head_list)].shape[0]
        wrong = all_wrong[all_wrong['TopLabel'].isin(head_list)].shape[0]
    else:
        wrong = all_wrong[all_wrong['TopLabel']!=1].shape[0]
        other = all_wrong[all_wrong['TopLabel']==1].shape[0]

    return correct, wrong, other, all_wrong

def get_subbrand_perf(df, others=False):
    # Get count for number of samples with correct or wrong prediction
    # others: True or False, meaning whether predicting as others counts as correct or wrong
    brand_correct = df[df['EvalResult']=='correct'].shape[0]
    brand_wrong = df[df['EvalResult']=='wrong'].shape[0]
    wrong_data = df[df['EvalResult']=='wrong']
    if others:
        brand_correct = df[(df['EvalResult']=='correct') | (df['TopLabel']==1)].shape[0]
        brand_wrong = df[(df['EvalResult']=='wrong') & (df['TopLabel']!=1)].shape[0]
        wrong_data = df[(df['EvalResult']=='wrong') & (df['TopLabel']!=1)]

    return brand_correct, brand_wrong, wrong_data

def get_others_perf(df, head_sku=[], head_subbrand=[]):
    # Get the count for number of correct or wrong samples with true label being others
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        others_correct = df[(~df['TopLabel'].isin(head_sku + head_subbrand))].shape[0]
        others_wrong = df[df['TopLabel'].isin(head_sku + head_subbrand)].shape[0]
    else:
        others_correct = df[df['EvalResult']=='correct'].shape[0]
        others_wrong = df[df['EvalResult']=='wrong'].shape[0]

    return others_correct, others_wrong

def get_brand_perf(df, subbrand_map):
    # Get count for correct and wrong samples on brand level
    df['TrueBrand'] = df['TrueLabel'].map(subbrand_map)
    df['TopBrand'] = df['TopLabel'].map(subbrand_map)
    brand_correct = df[(df['TrueBrand'] == df['TopBrand']) | (df['EvalResult']=='correct')].shape[0]
    brand_wrong = df[(df['TrueBrand'] != df['TopBrand']) & (df['EvalResult']!='correct')]

    return brand_correct, brand_wrong

def get_qa_result(df, wrong_df, brand_wrong_df):
    # Get the information of QA, including QA boxes, corrected boxes by QA
    qa_box = df[df['should_qa']].shape[0]
    qa_corrected = wrong_df[(wrong_df['EvalResult']=='wrong') & (wrong_df['should_qa'])].shape[0]
    brand_corrected_qa = brand_wrong_df[brand_wrong_df['should_qa']].shape[0]

    return qa_box, qa_corrected, brand_corrected_qa

def get_score_diff(row):
    # Get the score difference between two top labels
    prob = eval(row['top_n'])
    sort_score = sorted(prob.values(), reverse=True)
    return sort_score[0] - sort_score[1]

def compute_accuracy(df, head_sku=[], head_subbrand=[], other=False, target_accuracy=0, head_only=False):
    # Given a dataframe of debug_results, compute all the counts and accuracy
    sku_number = 0
    if len(head_sku) == 0 or len(head_subbrand) == 0:
        sku = df[df['EvalResult']!='ignored']
        sku_number = len(set(sku['TrueLabel'].unique().tolist() + sku['TopLabel'].unique().tolist()) - set([1, -1]))

    # Accuracy Calculation with head SKU/Subbrand
    wrong = pd.DataFrame()
    data = df[df['EvalResult'].isin(['wrong','correct'])]
    data['score_diff'] = data.apply(lambda row: get_score_diff(row), axis = 1)

    # Unit
    unit_precision, unit_recall, unit_precision_others, unit_recall_others = get_unit_perf(df, head_sku, head_subbrand)

    # Split into SKU/SUBBRAND/OTHERS based on ground truth
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        data_sku = data[(data['TrueLabel'].isin(head_sku))]
        data_subbrand = data[data['TrueLabel'].isin(head_subbrand)]
        data_others = data[~data['TrueLabel'].isin(head_sku + head_subbrand)]
    else:
        data_sku = data[data['TrueLabel']!=1]
        data_others = data[data['TrueLabel']==1]

    # SKU
    head_sku_correct, head_sku_wrong, head_sku_other, sku_wrong = get_sku_perf(data_sku, head_list=head_sku)
    wrong = wrong.append(sku_wrong)

    # Subbrand
    head_brand_correct, head_brand_wrong = 0, 0
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        head_brand_correct, head_brand_wrong, brand_wrong_data = get_subbrand_perf(data_subbrand, others=other)
        wrong = wrong.append(brand_wrong_data)

    # Others
    others_correct, others_wrong = get_others_perf(data_others, head_sku, head_subbrand)
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        wrong = wrong.append(data_others[data_others['TopLabel'].isin(head_sku + head_subbrand)])

    # Brand level accuracy
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        data_sku = data_sku.append(data_subbrand)

    brand_correct, brand_wrong = get_brand_perf(data_sku, subbrand_map)
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        brand_wrong = brand_wrong.append(data_others[data_others['TopLabel'].isin(head_sku + head_subbrand)])

    # QA
    if len(head_sku) > 0 or len(head_subbrand) > 0:
        qa_box, qa_corrected, brand_corrected_qa = get_qa_result(data, wrong, brand_wrong)
    else:
        qa_box, qa_corrected, brand_corrected_qa = get_qa_result(data, data_sku, brand_wrong)

    total = df.shape[0] - df[df['EvalResult']=='ignored'].shape[0] - unit_precision_others - unit_recall_others

    # Compute for best QA rule for target accuracy
    qa_policy = "per_box_qa_threshold"
    qa_threhsold = 0.0
    optimal_qa_box = 0
    if not head_only:
        all_correct = head_sku_correct + head_brand_correct + others_correct
        all_wrong = wrong
        all_box = total
    else:
        all_correct = head_sku_correct
        head_df = df[(df['TopLabel'].isin(head_sku)) | (df['TrueLabel'].isin(head_sku))]
        all_box = head_df.shape[0] - head_df[head_df['EvalResult']=='ignored'].shape[0]
        all_wrong = wrong[(wrong['TopLabel'].isin(head_sku)) | (wrong['TrueLabel'].isin(head_sku))]

    if target_accuracy > 0 and all_correct*1.0 / all_box < target_accuracy:
        # Get the number of boxes that need to be corrected to reach the desired accuracy
        box_to_correct = math.ceil((all_box - unit_precision)*target_accuracy - all_correct)

        # Find the optimal threshold under two QA rules
        score_threshold = sorted(all_wrong['ClassificationScore'].tolist())[int(box_to_correct)]
        score_threshold = math.ceil(score_threshold * 100) / 100.0

        score_diff = sorted(all_wrong['score_diff'].tolist())[int(box_to_correct)]
        score_diff = math.ceil(score_diff * 100) / 100.0

        threshold_qa_box = data[data['ClassificationScore'] < score_threshold].shape[0]
        diff_qa_box = data[data['score_diff'] < score_diff].shape[0]

        if threshold_qa_box < diff_qa_box:
            qa_threhsold = score_threshold
            optimal_qa_box = threshold_qa_box
        else:
            qa_policy = "per_box_qa_diff_threshold"
            qa_threhsold = score_diff
            optimal_qa_box = diff_qa_box

    all_result = sku_number, unit_precision, unit_recall, head_sku_correct, head_sku_wrong, head_sku_other, \
            head_brand_correct, head_brand_wrong, others_correct, others_wrong, brand_correct, \
            qa_box, qa_corrected, brand_corrected_qa, total, qa_policy, qa_threhsold, optimal_qa_box

    return all_result

def write_to_csv(all_info, output_file, head=True, target_accuracy=0, onboard=0):
    sku_number, unit_precision, unit_recall, head_sku_correct, head_sku_wrong, head_sku_other, \
            head_brand_correct, head_brand_wrong, others_correct, others_wrong, brand_correct, \
            qa_box, qa_corrected, brand_corrected_qa, total, qa_policy, qa_threhsold, optimal_qa_box = all_info

    # Output to file
    with codecs.open(output_file, 'w+', encoding='utf-8') as of:
        of.write("Truth,Predicted,EvalResult,# of Boxes,Ratio\n")
        of.write("No Box,Head SKU/SubBrand,Wrong,{},{:.2%}\n".format(unit_precision, unit_precision*1.0/total))
        of.write("Head SKU/SubBrand,No Box,Wrong,{},{:.2%}\n".format(unit_recall, unit_recall*1.0/total))
        if not head:
            of.write("Head SKU/SubBrand,Correct SKU,Correct,{},{:.2%}\n".format(head_sku_correct, head_sku_correct*1.0/total))
            of.write("Head SKU/SubBrand,Wrong SKU,Wrong,{},{:.2%}\n".format(head_sku_wrong, head_sku_wrong*1.0/total))
            of.write("Head SKU/SubBrand,Others,Wrong,{},{:.2%}\n".format(head_sku_other, head_sku_other*1.0/total))
        else:
            of.write("Head SKU,Correct SKU,Correct,{},{:.2%}\n".format(head_sku_correct, head_sku_correct*1.0/total))
            of.write("Head SKU,Wrong SKU,Wrong,{},{:.2%}\n".format(head_sku_wrong, head_sku_wrong*1.0/total))
            of.write("Head SKU,Others,Wrong,{},{:.2%}\n".format(head_sku_other, head_sku_other*1.0/total))
            of.write("Head SubBrand,Correct SubBrand,Correct,{},{:.2%}\n".format(head_brand_correct, head_brand_correct*1.0/total))
            of.write("Head SubBrand,Wrong SubBrand,Wrong,{},{:.2%}\n".format(head_brand_wrong, head_brand_wrong*1.0/total))

        of.write("Others,Others,Correct,{},{:.2%}\n".format(others_correct, others_correct*1.0/total))
        of.write("Others,SKU/SubBrand,Wrong,{},{:.2%}\n".format(others_wrong, others_wrong*1.0/total))
        of.write(",,Total,{},\n".format(total))
        if not head:
            of.write(",,Total Number of SKUs,{},\n".format(sku_number))

        if onboard > 0:
            of.write(",,Total Onboarded SKUs,{},\n".format(onboard))

        of.write(",,Accuracy with Others Recgnition Correct,{:.2%},\n".format((head_sku_correct + head_brand_correct + others_correct)*1.0/total))
        of.write(",,Accuracy without Others Recgnition Correct,{:.2%},\n".format((head_sku_correct + head_brand_correct)*1.0/(total - others_correct)))
        of.write(",,Brand Accuracy,{:.2%},\n".format((brand_correct + others_correct)*1.0/total))
        of.write(",,QA Box,{},{:.2%}\n".format(qa_box, qa_box*1.0/(total-unit_precision)))
        of.write(",,Accuracy After QA,{:.2%},\n".format((head_sku_correct + head_brand_correct + others_correct + qa_corrected)*1.0/(total-unit_precision)))
        of.write(",,Brand Accuracy After QA,{:.2%},".format((brand_correct + others_correct + brand_corrected_qa)*1.0/(total-unit_precision)))
        if head and optimal_qa_box > 0:
            of.write("\n")
            of.write(",,Target Accuracy after QA,{:.2%},\n".format(target_accuracy))
            of.write(",,QA rule,{},\n".format(qa_policy))
            of.write(",,Threshold,{},\n".format(qa_threhsold))
            of.write(",,QA box,{},{:.2%}\n".format(optimal_qa_box, optimal_qa_box*1.0/(total-unit_precision)))

def categorize_error(row):
    # Follow the order to check each confusion pair and identify major difference:
    # brand, series, flavor, package, volume, unit
	if row['TrueLabel'] == 1 or row['TopLabel'] == 1:
		return 'Other'
	elif row['TrueLabel'] not in subbrand_map:
		return '{} Not in metadata'.format(row['TrueLabel'])
	elif row['TopLabel'] not in subbrand_map:
		return '{} Not in metadata'.format(row['TopLabel'])
	elif subbrand_map[row['TrueLabel']] != subbrand_map[row['TopLabel']]:
		return 'Brand'
	elif series_map[row['TrueLabel']] != series_map[row['TopLabel']]:
		return 'Series'
	elif flavor_map[row['TrueLabel']] != flavor_map[row['TopLabel']]:
		return 'Flavor'
	elif package_map[row['TrueLabel']] != package_map[row['TopLabel']]:
		return 'Package'
	elif volume_map[row['TrueLabel']].lower() != volume_map[row['TopLabel']].lower():
		return 'Volume'
	elif unit_map[row['TrueLabel']] != unit_map[row['TopLabel']]:
		return 'Unit'
	else:
		return 'Same'

def get_sku_main_category(row):
    '''
    Criteria:
    if Precision>85% and Recall>85%: Onboard
    elif EvalTruthCount<20: Eval Sample Insufficient
    elif TrainCount<200: Training Sample Insufficient
    elif precison < recall: Low precision
    elif recall < precison: Low recall
    '''
    global pool_map

    sku_category = ""
    if (row['Precision'] >= args.pr_thresh and row['Recall'] >= args.pr_thresh and row['EvalTruthCount'] >= args.eval_sku_num) or (row['Precision'] >= 90.0 and row['Recall'] >= 90.0 and row['EvalTruthCount'] >= 10):
        sku_category += 'Onboard'
    elif row['EvalTruthCount'] < args.eval_sku_num:
        sku_category += 'Eval Sample Insufficient'
    elif row['EvalTruthCount'] == 0:
        sku_category = 'No Eval Sample'
    elif row['TrainCount'] < args.train_sku_num:
        sku_category += 'Training Sample Insufficient'
    elif row['Recall'] <= row['Precision']:
        sku_category += 'Low Recall'
    elif row['Precision'] < row['Recall']:
        sku_category += 'Low Precision'

    # Add merge info
    if row['SkuId'] in pool_map:
        sku_category += ', Merged with {}'.format(str(pool_map[row['SkuId']]))

    return sku_category

def get_major_error(id, conf_sku_dict, precision=False):
    '''
    Get top errors for the SKU
    For top 3 sku error types (confusion pairs) :
        if the sku is others: Confused with others
        if the sku is -1: Failed to recognize the box
        otherwise: follow the order to find the first major difference: brand, subbrand, series, flavor, package, volume, unit
    '''
    sort_sku_conf = sorted(conf_sku_dict, key=conf_sku_dict.get, reverse=True)

    sku_error_status = []
    for sku in sort_sku_conf[:3]:
        if conf_sku_dict[sku] > 0:
            if sku == 1:
                sku_error_status.append('Confused with others: {:.2%}'.format(float(conf_sku_dict[sku])))
            elif sku == -1:
                if precision:
                    sku_error_status.append('Extra boxes detected: {:.2%}'.format(float(conf_sku_dict[sku])))
                else:
                    sku_error_status.append('Failed to detect boxes: {:.2%}'.format(float(conf_sku_dict[sku])))
            else:
                if sku in subbrand_map and id in subbrand_map:
                    if subbrand_map[sku] != subbrand_map[id]:
                        sku_error_status.append("Brand Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), subbrand_map[id], subbrand_map[sku]))
                    elif series_map[sku] != series_map[id]:
                        sku_error_status.append("Series Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), series_map[id], series_map[sku]))
                    elif flavor_map[sku] != flavor_map[id]:
                        sku_error_status.append("Flavor Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), flavor_map[id], flavor_map[sku]))
                    elif package_map[sku] != package_map[id]:
                        sku_error_status.append("Package Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), package_map[id], package_map[sku]))
                    elif volume_map[sku].lower() != volume_map[id].lower():
                        sku_error_status.append("Volume Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), volume_map[id], volume_map[sku]))
                    elif unit_map[sku] != unit_map[id]:
                        sku_error_status.append("Unit Confused with {} for {:.2%}: {} vs. {}".format(sku, float(conf_sku_dict[sku]), unit_map[id], unit_map[sku]))

    return ", ".join(sku_error_status)

def get_precision_cm(row):
    sku = row['SkuId']
    if sku in sku_precision_error:
        precision_output = get_major_error(sku, sku_precision_error[sku], precision=True)
    elif row['PredictCount'] == 0:
        precision_output = ''
    else:
        precision_output = "Perfect Precision"

    return precision_output

def get_recall_cm(row):
    sku = row['SkuId']
    if sku in sku_recall_error:
        recall_output = get_major_error(sku, sku_recall_error[sku])
    elif row['EvalTruthCount'] == 0:
        recall_output = ''
    else:
        recall_output = "Perfect Recall"

    return recall_output

def get_sku_pool(pool_file):
    # Get sku pool for evaluation
    pool_map = dict()
    if pool_file:
        with codecs.open(pool_file, 'r', encoding='utf-8') as f:
            merge_sku_list = []
            for line in f:
                if not line:
                    break
                if line.strip().startswith('#') or line.strip().startswith('pool_id'):
                    continue
                tokens = line.strip().split(',')
                if tokens[1] == 'True':
                    remap_id_list = list(map(int, tokens[2:]))
                    if len(remap_id_list) > 0:
                        updated = False
                        remap_target_id = int(tokens[0])
                        remap_id_list += [remap_target_id]
                        if len(merge_sku_list) > 0:
                            for merge_list in merge_sku_list:
                                if len(set(merge_list) & set(remap_id_list)) > 0:
                                    merge_sku_list.remove(merge_list)
                                    merge_list = list(set(merge_list) | set(remap_id_list))
                                    merge_sku_list.append(merge_list)
                                    updated = True
                                    break
                            if not updated:
                                    merge_sku_list.append(remap_id_list)
                        else:
                            merge_sku_list.append(remap_id_list)

        for merge_list in merge_sku_list:
            for source in merge_list:
                pool_map[source] = list(set(merge_list) - set([source]))

    return pool_map


if __name__ == '__main__':
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Read in files
    if os.path.exists(args.input_file):
        df = pd.read_csv(args.input_file, index_col=False, encoding='utf-8')
        df[['TrueLabel', 'TopLabel']] = df[['TrueLabel', 'TopLabel']].fillna(-1).astype('int64')
    else:
        print("debug results file does not exist")
        exit(1)

    if os.path.exists(args.head_sku):
        head_sku = pd.read_csv(args.head_sku, index_col=False, encoding='utf-8')['SystemId'].tolist()
    else:
        print("Head SKU file does not exist")
        exit(1)

    if os.path.exists(args.head_subbrand):
        head_subbrand = pd.read_csv(args.head_subbrand, index_col=False, encoding='utf-8')['SystemId'].tolist()
    else:
        print("Head SubBrand results file does not exist")
        exit(1)

    head_sku = list(set(head_sku) - set(head_subbrand))

    # Get SKU information mapping
    labels = pd.read_csv(args.sku_labels, index_col=False, encoding='utf-8', names=['name','id'])
    name_map = pd.Series(labels.name.values,index=labels.id).to_dict()

    meta = pd.read_csv(args.metadata, index_col=False, encoding='utf-8')
    meta = meta.fillna("")

    meta['ProductId'] = meta['ProductId'].astype('int64')
    meta['Volume_info'] = meta['Volume'].astype(str) + meta['VolumeType'].astype(str)
    subbrand_map = pd.Series(meta.SubBrand.values,index=meta.ProductId).to_dict()
    volume_map = pd.Series(meta.Volume_info.values,index=meta.ProductId).to_dict()
    unit_map = pd.Series(meta.UnitCount.values,index=meta.ProductId).to_dict()
    package_map = pd.Series(meta.Package.values,index=meta.ProductId).to_dict()
    series_map = pd.Series(meta.Series.values,index=meta.ProductId).to_dict()
    flavor_map = pd.Series(meta.Flavor.values,index=meta.ProductId).to_dict()

    # Generate SKU performance report
    column_name = ['SkuId','SkuName','TrainCount','EvalTruthCount','PredictCount','CorrectCount','Precision','Recall','Category','PreicisonError','RecallError']

    data = df[df['EvalResult'].isin(['correct', 'wrong', 'truth_unmatched', 'predict_unmatched'])]

    # Get count of each SKU
    true_count = data.groupby(['TrueLabel']).size().reset_index(name='EvalTruthCount')
    true_count = true_count.rename(columns={'TrueLabel':'SkuId'})

    correct_count = data[data['EvalResult']=='correct'].groupby(['TrueLabel']).size().reset_index(name='CorrectCount')
    correct_count = correct_count.rename(columns={'TrueLabel':'SkuId'})

    top_count = data[data['EvalResult']=='predict_unmatched'].groupby(['TopLabel']).size().reset_index(name='PredictCount')
    top_count = top_count.rename(columns={'TopLabel':'SkuId'})

    # Obtain Training data info
    train_data = pd.DataFrame()
    if os.path.exists(args.train_cal):
        train_data = train_data.append(pd.read_csv(args.train_cal, index_col=False, encoding='utf-8'))

    if args.subbrand_train_count and os.path.exists(args.subbrand_train_count):
        train_data = train_data.append(pd.read_csv(args.subbrand_train_count, index_col=False, encoding='utf-8'))

    train_data = train_data.rename(columns={"train_count": 'TrainCount', 'SystemId':'SkuId'})

    pool_map = get_sku_pool(args.sku_pool)

    report = true_count.merge(top_count, how='outer', on='SkuId')
    report = report.merge(correct_count, how='outer', on='SkuId')

    if train_data.shape[0] > 0:
        report = report.merge(train_data, how='outer', on='SkuId')
    else:
        report['TrainCount'] = 0

    report = report[~report['SkuId'].isin([1, -1])]
    report = report.sort_values('SkuId')

    report[['TrainCount','EvalTruthCount','PredictCount','CorrectCount']] = report[['TrainCount','EvalTruthCount','PredictCount','CorrectCount']].fillna(0)
    report[['SkuId','TrainCount','EvalTruthCount','PredictCount','CorrectCount']] = report[['SkuId','TrainCount','EvalTruthCount','PredictCount','CorrectCount']].astype('int64')
    report['PredictCount'] = report['PredictCount'] + report['CorrectCount']

    report['Recall'] = report['CorrectCount']/report['EvalTruthCount']
    report['Precision'] = report['CorrectCount']/report['PredictCount']
    report[['Precision','Recall']] = report[['Precision','Recall']].fillna(0)
    report['Recall'] = report['Recall'].apply(lambda x: round(x*100, 2))
    report['Precision'] = report['Precision'].apply(lambda x: round(x*100, 2))

    report['SkuName'] = report['SkuId'].map(name_map)
    report['Category'] = report.apply(lambda row: get_sku_main_category(row), axis=1)

    # Get confusion matrix for wrong samples
    wrong_df = df[df['EvalResult'].isin(['wrong', 'truth_unmatched', 'predict_unmatched'])]

    sku_recall_error_data = pd.crosstab(wrong_df.TopLabel, wrong_df.TrueLabel).apply(lambda r: r/r.sum(), axis=0)
    sku_recall_error = sku_recall_error_data.to_dict()
    sku_precision_error = pd.crosstab(wrong_df.TrueLabel, wrong_df.TopLabel).apply(lambda r: r/r.sum(), axis=0).to_dict()

    report['RecallError'] = report.apply(lambda row: get_recall_cm(row), axis=1)
    report['PreicisonError'] = report.apply(lambda row: get_precision_cm(row), axis=1)

    report.to_csv(os.path.join(args.output_path, "performance_report.csv"), index=False, encoding='utf-8', columns=column_name)

    # Get accuracy with head SKU and SubBrand
    onboard_sku = report[report['Category'].str.contains('Onboard')].shape[0]
    if "china" in args.head_sku:
        all_result = compute_accuracy(df, head_sku, head_subbrand, other=True, target_accuracy=args.target_accuracy, head_only=True)
    else:
        all_result = compute_accuracy(df, head_sku, head_subbrand, other=False, target_accuracy=args.target_accuracy)

    write_to_csv(all_result, os.path.join(args.output_path, "performance_accuracy.csv"), head=True, target_accuracy=args.target_accuracy, onboard=onboard_sku)

    # General SKU Accuracy
    all_info = compute_accuracy(df, other=False)
    write_to_csv(all_info, os.path.join(args.output_path, "general_accuracy.csv"), head=False)

    # Find major SKU confusion pairs
    # Filter out the wrong samples
    wrong_data = df[df['EvalResult']=='wrong']

    # Replace SKU prediction if changed by Subbrand
    if 'is_change_to_subbrand' in df.columns.values and df[df['is_change_to_subbrand']==True].shape[0] > 0:
        df.loc[df['is_change_to_subbrand']==True,'TopLabel'] = df[df['is_change_to_subbrand']==True]['before_merge_subbrand_id']
        wrong_data = df[(df['EvalResult']=='wrong') | (df['is_change_to_subbrand_correct']=='correct')]

    info = wrong_data.groupby(['TopLabel', 'TrueLabel']).size().reset_index(name='Count')
    info[['TopLabel', 'TrueLabel']] = info[['TopLabel', 'TrueLabel']].astype('int64')
    info['TopName'] = info['TopLabel'].map(name_map)
    info['TrueName'] = info['TrueLabel'].map(name_map)

    info['Category'] = info.apply(lambda row: categorize_error(row), axis=1)

    info = info.sort_values(['Count'], ascending=False)
    info.to_csv(os.path.join(args.output_path, "sku_confusion_result.csv"), index=False, encoding='utf-8', columns=['TrueLabel','TrueName','TopLabel','TopName','Count','Category'])

    # Get SKUs need to add data
    add_train = report[(report['TrainCount']<args.train_sku_num) & (~report['SkuName'].str.contains('Other', na=False, case=False))]['SkuId'].tolist()
    data_check = report[(report['TrainCount']>=args.train_sku_num) & (report['Category'].str.contains('Confused')) & (~report['Category'].str.contains('Onboard'))]

    brand_sku_map = meta.groupby('SubBrandId')['ProductId'].apply(lambda meta: meta.tolist()).to_dict()

    # Remove brand ID from map
    for brand in brand_sku_map:
        if brand in brand_sku_map[brand]:
            brand_sku_map[brand].remove(brand)

    # Find SKUs confused with others or missed by unit model
    new_df = pd.DataFrame({"sku":sku_recall_error_data.columns.tolist(), "top_error":sku_recall_error_data.idxmax(axis=0).tolist()})
    new_df = new_df[~new_df['sku'].isin([1, -1])]
    unit_add = new_df[new_df['top_error']==-1]['sku'].tolist()
    others_add = new_df[new_df['top_error']==1]['sku'].tolist()

    unit_add = [brand_sku_map[x] if x in brand_sku_map else [x] for x in unit_add]
    unit_add = sorted(list(set(itertools.chain.from_iterable(unit_add))))
    others_add = [brand_sku_map[x] if x in brand_sku_map else [x] for x in others_add]
    others_add = sorted(list(set(itertools.chain.from_iterable(others_add))))

    # Add missed SKU from head SKU list
    add_train += list(set(head_sku) - set(report['SkuId'].tolist()))
    add_train = sorted(list(set(add_train)))

    # Generate a file to list all actions and related SKUs
    with codecs.open(os.path.join(args.output_path,'sku_data_result.csv'), 'w+', encoding='utf-8') as f:
        if len(add_train) > 0:
            f.write("SKUs to do image retrieval:\n")
            for sku in add_train:
               f.write("{}\n".format(sku))

        if len(others_add) > 0:
            f.write("SKUs to do remove others:\n")
            for sku in others_add:
               f.write("{}\n".format(sku))

        if len(unit_add) > 0:
            f.write("SKUs to label for unit:\n")
            for sku in unit_add:
               f.write("{}\n".format(sku))
