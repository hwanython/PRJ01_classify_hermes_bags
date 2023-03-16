import os
import glob
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from PRJ01_classify_hermes_bags.configs.CLASSES import CLASSES

parser = argparse.ArgumentParser(description='make data table .csv')

parser.add_argument('--root', default=r'../datasets/raws')
parser.add_argument('--save_dir', default=r'../datasets/csv')
parser.add_argument('--debug', default=False)
args = parser.parse_args()

if __name__ == '__main__':

    class_folder = os.listdir(args.root)

    table = {
        'labels': [],
        'filepath': [],
    }
    # print(CLASSES.keys())
    # print(class_folder)
    for idx, label in enumerate(class_folder):
        data_list = glob.glob(os.path.join(args.root, label, '*', '*.jpg'))
        bag_label = [CLASSES[label]] * len(data_list)
        table['labels'].extend(bag_label)
        table['filepath'].extend(data_list)

    data = pd.DataFrame(table)

    X_train, X_valid_1, y_train, y_valid_1 = train_test_split(data['filepath'], data['labels'],
                                                              test_size=0.2, shuffle=True, random_state=8)
    # train, valid, test = StratifiedDataSplit(dataframe=df).train_val_test()
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_1, y_valid_1,
                                                        test_size=0.5, shuffle=True, random_state=5)
    for idx in range(6):
        label = idx + 1
        train_label_num = list(y_train.values).count(label)
        valid_label_num = list(y_valid.values).count(label)
        test_label_num = list(y_test.values).count(label)
        if args.debug > 0:
            print(
                f"=====================================================\nDataset Split Label: {label}\n  train: {train_label_num}\n "
                f" valid: {valid_label_num}\n  test: {test_label_num}\n=====================================================")

    # save file
    train_list = [X_train, y_train]
    valid_list = [X_valid, y_valid]
    test_list = [X_test, y_test]

    train_df = pd.DataFrame(train_list).transpose().reset_index()
    valid_df = pd.DataFrame(valid_list).transpose().reset_index()
    test_df = pd.DataFrame(test_list).transpose().reset_index()

    train_df.to_csv(os.path.join(args.save_dir, 'train.csv'), sep=',', na_rep='NaN')
    valid_df.to_csv(os.path.join(args.save_dir, 'valid.csv'), sep=',', na_rep='NaN')
    test_df.to_csv(os.path.join(args.save_dir, 'test.csv'), sep=',', na_rep='NaN')

    print("Complete!")
    print(
        f"=====================================================\nDataset Split Total: {len(data['labels'])}\n  train: {len(X_train)}\n "
        f" valid: {len(X_valid)}\n  test: {len(X_test)}\n=====================================================")

