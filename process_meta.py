import json
import math
import pickle
import random

from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from skmultilearn.embedding import EmbeddingClassifier
from scipy.sparse import lil_matrix
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN


def get_tasks():
    # r = requests.get('https://codeforces.com/api/problemset.problems')
    # problems = get_data('problemset.problems')
    with open('tasks_info.json') as data_file:
        tasks = json.load(data_file)
        return tasks['result']['problems']


def get_meta():
    with open('extracted_meta.json') as meta_file:
        return json.load(meta_file)


def unify_names(meta):
    def unify_name(name):
        return name.lower().replace('_', '')

    for key, value in meta.items():
        for item_key in 'variables', 'function_names':
            meta[key][item_key] = [unify_name(x) for x in value[item_key]]
    return meta


def group_meta_by_tasks(meta):
    meta_by_task = defaultdict(lambda: defaultdict(list))
    for key, value in meta.items():
        parts = key.split('/')
        for item_key in 'variables', 'function_names':
            meta_by_task[parts[1], parts[2]][item_key] += value[item_key]
    return meta_by_task


def tfidf(tag_meta):
    result = defaultdict(dict)
    for item_key in 'variables', 'function_names':
        idf = {}
        total = len(tag_meta)
        all_words = set(sum([x[item_key] for x in tag_meta.values()], []))
        word_frequency = Counter(
            sum([list(set(x[item_key])) for x in tag_meta.values()], [])
        )
        for word in all_words:
            idf[word] = math.log(total * 1. / word_frequency[word])

        for tag in tag_meta:
            word_count = Counter(tag_meta[tag][item_key])
            tfidf = {}
            for word in word_count:
                tf = word_count[word]
                tfidf[word] = float(tf) * idf[word]

            result[tag][item_key] = sorted(tfidf.items(), key=lambda x: -x[1])
    return result


def process(tasks, meta, debug=False):
    tag_meta = defaultdict(lambda: defaultdict(list))
    for problem in tasks:
        tags = problem['tags']
        contest_id = str(problem['contestId'])
        index = problem['index']
        for item_key in 'variables', 'function_names':
            for tag in tags:
                tag_meta[tag][item_key] += meta[contest_id, index][item_key]

    tfidf_scores = tfidf(tag_meta)

    if debug:
        for tag in tag_meta:
            print('tag:', tag)
            for item_key in 'variables', 'function_names':
                print(tag, item_key, ':')
                for ind, i in enumerate(tfidf_scores[tag][item_key][:20]):
                    print(ind, i[1], i[0])
                print
            print
    return tfidf_scores


TEST_PART = 0.03


def split_on_train_test(tasks, meta):
    train_tasks = []
    test_tasks = []
    train_meta = {}
    test_meta = {}
    random.shuffle(tasks)
    tasks = [task for task in tasks if task['tags']]
    test_size = int(len(tasks) * TEST_PART)

    big_enough_tasks = []
    small_tasks = []
    for task in tasks:
        if len(meta[str(task['contestId']), task['index']]['variables']) > 800:
            big_enough_tasks.append(task)
        else:
            small_tasks.append(task)

    for curr_tasks, curr_meta, curr_src_tasks in (
        (test_tasks, test_meta, big_enough_tasks[:test_size]),
        (train_tasks, train_meta, big_enough_tasks[test_size:] + small_tasks),
    ):
        for task in curr_src_tasks:
            curr_tasks.append(task)
            curr_meta[str(task['contestId']), task['index']] = meta[str(task['contestId']), task['index']]
    return train_tasks, test_tasks, train_meta, test_meta


SCORE_LIMIT = 2.0


def calc_reversed_tfidf(tfidf):
    reversed_tfidf = defaultdict(lambda: defaultdict(list))
    for item_key in 'variables', 'function_names':
        for tag in tfidf:
            for var, score in tfidf[tag][item_key]:
                if score > SCORE_LIMIT:
                    reversed_tfidf[item_key][var].append({'score': score, 'tag': tag})
    return reversed_tfidf


def predict(tasks, meta, reversed_tfidf, debug=False):
    result = []
    for problem in tasks:
        tag_scores = defaultdict(float)
        contest_id = str(problem['contestId'])
        index = problem['index']
        for item_key in 'variables', 'function_names':
            for var in meta[contest_id, index][item_key]:
                for score in reversed_tfidf[item_key][var]:
                    tag_scores[score['tag']] += score['score']
        real_tags = problem['tags']
        tag_scores_items = tag_scores.items()
        tag_scores_items.sort(key=lambda x: -x[1])
        result.append({
            'contestId': problem['contestId'],
            'index': problem['index'],
            'real_tags': real_tags,
            'predicted': tag_scores_items,
        })
    with open('predict_result_full.json', 'w') as ou:
        json.dumps(result, ou)
    return result


def calc_top1(results):
    good = 0
    for item in results:
        good += item['predicted'][0][0] in item['real_tags']
    return float(good) / len(results)


def calc_top_n(results, n):
    good = 0
    for item in results:
        good += (
                float(len(set([x[0] for x in item['predicted'][:n]]).intersection(item['real_tags']))) /
                min(n, len(item['real_tags']))
        )
    return float(good) / len(results)


def calcScores(results):
    for i in (1, 5, 10):
        print('top n {} score:'.format(i), calc_top_n(results, i))


def train_bm25(tasks, meta):
    import gensim.summarization.bm25 as bm25
    tag_meta = defaultdict(list)
    for problem in tasks:
        tags = problem['tags']
        contest_id = str(problem['contestId'])
        index = problem['index']
        for item_key in 'variables', 'function_names':
            for tag in tags:
                tag_meta[tag] += ['{}{}'.format(item_key[0], x) for x in meta[contest_id, index][item_key]]
    keys = tag_meta.keys()
    corpus = tag_meta.values()
    bm25_object = bm25.BM25(corpus)
    return bm25_object, keys


def predict_bm25(tasks, meta, bm25_object, keys):
    result = []
    for problem in tasks:
        contest_id = str(problem['contestId'])
        index = problem['index']
        request = []
        for item_key in 'variables', 'function_names':
            request += ['{}{}'.format(item_key[0], x) for x in meta[contest_id, index][item_key]]
        scores = bm25_object.get_scores(request)
        real_tags = problem['tags']
        tag_scores_items = sorted(zip(keys, scores), key=lambda x: -x[1])
        result.append({
            'contestId': problem['contestId'],
            'index': problem['index'],
            'real_tags': real_tags,
            'predicted': tag_scores_items,
        })
    return result


def transform_data(tasks, meta, all_tags):
    x = []
    y = []
    for ind, problem in enumerate(tasks):
        if not isinstance(problem, str):
            tags = set(problem['tags'])
            contest_id = str(problem['contestId'])
            index = problem['index']
            current_x = []
            for item_key in 'variables', 'function_names':
                current_x.append(' '.join(['{}{}'.format(item_key[0], value) for value in meta[contest_id, index][item_key]]))
            x.append(' '.join(current_x))
            y.append([int(tag in tags) for tag in all_tags])
        else:
            x.append(problem)
            y.append([int(tag in meta[ind]) for tag in all_tags])

    return x, y


def mlknn(train_tasks, test_tasks, train_meta, test_meta, x_spl, y_spl):

    with open('predictions_new_OpenNetworkEmbedder.pickle', 'rb') as f:
        data = pickle.load(f)
        predictions_new = data['predictions']
        all_tags = data['all_tags']
    all_tags = []
    for problem in train_tasks:
        all_tags += problem['tags']
    for problem in test_tasks:
        all_tags += problem['tags']
    all_tags = list(set(all_tags))

    x_train, y_train = transform_data(train_tasks, train_meta, all_tags)
    x_test, y_test = transform_data(test_tasks, test_meta, all_tags)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    vectorizer.fit(x_test)
    x_train_transformed = vectorizer.transform(x_train)
    x_test_transformed = vectorizer.transform(x_test)


    x_train = lil_matrix(x_train_transformed).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test_transformed).toarray()
    y_test = lil_matrix(y_test).toarray()

    from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
    from sklearn.manifold import SpectralEmbedding
    from sklearn.ensemble import RandomForestRegressor
    from skmultilearn.adapt import MLkNN
    for n in (10000000,):
        clf = EmbeddingClassifier(
            SKLearnEmbedder(SpectralEmbedding(n_components=10)),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5)
        )

        clf.fit(x_train[:n], y_train[:n])

        predictions = clf.predict(x_test)

        from sklearn.metrics import accuracy_score
        print("Accuracy = ", accuracy_score(y_test, predictions))


    graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
    openne_line_params = dict(batch_size=1000, order=3)
    embedder = OpenNetworkEmbedder(
        graph_builder,
        'LINE',
        dimension=5 * y_train.shape[1],
        aggregation_function='add',
        normalize_weights=True,
        param_dict=openne_line_params
    )

    clf = EmbeddingClassifier(
        embedder,
        RandomForestRegressor(n_estimators=10),
        MLkNN(k=5)
    )

    clf.fit(x_train, y_train)
    with open('model.pickle', 'wb') as f:
        pickle.dump({'all_tags': all_tags, 'model': clf}, f)

    predictions_new = clf.predict(x_test)


    # with open('predictions_new_OpenNetworkEmbedder.pickle', 'wb') as f:
    #     pickle.dump({'all_tags': all_tags, 'predictions': predictions_new}, f)
    # with open('predictions_new_OpenNetworkEmbedder.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #     predictions_new = data['predictions']
    #     all_tags_pickle = data['all_tags']
    #     # permutate y to be consistent with
    #     y_test_new = y_test
    # return predictions_new, all

    for predicted, real in zip(predictions_new.toarray(), y_test):
        pt = []
        rt = []
        # for ap, aap in zip(predicted, all_tags_pickle):
        #     if ap:
        #         pt.append(aap)
        # for ar, aa in zip(real, all_tags):
        #     if ar:
        #         rt.append(aa)

        for ap, ar, aa in zip(predicted, real, all_tags):
            if ap:
                pt.append(aa)
            if ar:
                rt.append(aa)
        print('predicted', pt)
        print('real', rt)
        print('-' * 20, '\n')

    from sklearn.metrics import accuracy_score
    print("Accuracy = ", accuracy_score(y_test, predictions_new))
    print("Accuracy = ", accuracy_score(predictions_new, y_test))


def test_spl(test_tasks, test_meta, meta):
    tasks = set('{}/{}'.format(task['contestId'], task['index']) for task in test_tasks)
    spl_meta = [x for x in meta.items() if '/'.join(x[0].split('/')[1:3]) in tasks]
    print(len(meta), len(spl_meta))
    x = []
    y = []
    for item in spl_meta:
        task_contest = item[0].split('/')[1]
        task_index = item[0].split('/')[2]

        for t in test_tasks:
            if t['contestId'] == task_contest and t['index'] == task_index:
                break
        tags = t['tags']
        l = []
        for item_key in 'variables', 'function_names':
            l += ['{}{}'.format(item_key[0], x) for x in item[1][item_key]]
        if len(l) > 20:
            x.append(' '.join(l))
            y.append(tags)
    return x, y


def main():
    random.seed(41)
    tasks = get_tasks()
    meta = get_meta()

    groupped_meta = group_meta_by_tasks(unify_names(meta))
    train_tasks, test_tasks, train_meta, test_meta = split_on_train_test(tasks, groupped_meta)
    x_spl, y_spl = test_spl(test_tasks, test_meta, meta)
    tfidf_scores = process(tasks, groupped_meta, debug=True)
    reversed_tfidf = calc_reversed_tfidf(tfidf_scores)
    tfidf_results = predict(test_tasks, test_meta, reversed_tfidf, debug=True)
    print('tfidf')
    calcScores(tfidf_results)

    bm25_object, keys = train_bm25(train_tasks, train_meta)
    bm25_results = predict_bm25(test_tasks, test_meta, bm25_object, keys)
    print('bm25')
    calcScores(bm25_results)

    predicted = mlknn(train_tasks, test_tasks, train_meta, test_meta, x_spl, y_spl)


if __name__ == '__main__':
    main()
