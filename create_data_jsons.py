import json


def create_train_test_jsons():
    bad, average, good = [], [], []

    with open('scores.txt') as sc:
        scores = sc.readlines()

    for line in scores:
        filename, label = line.rstrip('\n').split(',')
        if label == 'bad':
            bad.append(filename)
        elif label == 'average':
            average.append(filename)
        elif label == 'good':
            good.append(filename)

    # print(len(bad), len(average), len(good))  # all classes have 334 samples

    class_sizes = (len(bad) + len(average) + len(good)) // 3  # int
    split_size = 0.9
    train_size = int(len(bad) * split_size)

    train = {'filepath': [], 'label': []}
    test = {'filepath': [], 'label': []}

    for i in range(class_sizes):
        if i < train_size:
            train['filepath'].append(bad[i])
            train['label'].append('bad')
            train['filepath'].append(average[i])
            train['label'].append('average')
            train['filepath'].append(good[i])
            train['label'].append('good')
        else:
            test['filepath'].append(bad[i])
            test['label'].append('bad')
            test['filepath'].append(average[i])
            test['label'].append('average')
            test['filepath'].append(good[i])
            test['label'].append('good')

    with open('train_data.json', 'w') as fp:
        json.dump(train, fp)
    with open('test_data.json', 'w') as fp:
        json.dump(test, fp)
