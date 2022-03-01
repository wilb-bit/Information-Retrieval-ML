def get_ndcg_score(txt):
    for ln in txt.split('\n'):
        ln = ln.strip()
        fields = ln.split('\t')
        metric = fields[0].strip()
        if metric == 'ndcg':
            return ln


def run_trec(this_os, qrel_file, run_file, verbose):
    if this_os == 'Linux':
        tbin = './trec_eval.linux'
    elif this_os == 'Windows':
        tbin = 'trec_eval.exe'
    elif this_os == 'Darwin':
        tbin = './trec_eval.osx'
    else:
        print('OS is not known')

    try:
        args = (tbin, "-m", "all_trec",
                qrel_file, run_file)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        txt = output.decode()
        if verbose == True:
            print(txt)
        # else:
        # print (get_ndcg_score(txt))
    except Exception as e:
        print('[ERROR]: subprocess failed')
        print('[ERROR]: {}'.format(e))
    return get_ndcg_score(txt)


# NOTE: This is splitting on a space. Your output should use a tab
# So -- x,y,z = ln.split('\t')
def read_sort_run(run_file):
    qdic = {}
    lines = []
    with open(run_file, 'r') as f:
        for ln in f:
            ln = ln.strip()
            # print (ln)
            x, y, z = ln.split('\t')
            if x in qdic:
                qdic[x].append((y, float(z)))
            else:
                qdic[x] = []
                qdic[x].append((y, float(z)))

    rank = 1
    for k, v in qdic.items():
        v.sort(key=lambda x: x[1], reverse=True)
        rank = 1
        for a, b in v:
            out = str(k) + ' Q0 ' + a + ' ' + str(rank) + ' ' + str(b) + ' e76767'
            lines.append(out)
            rank += 1
    return '\n'.join(lines)


def parameter_sweep():
    train_all = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')

    X = train_all
    y = train_all["Label"]
    groups = train_all['#QueryID']
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=4)
    t = np.zeros((1296, 5))
    i = 0

    #k fold with 10 splits. the averaged ndcg score across the 10 folds will be used to find the best hyperparameters for the model
    for train_idxs, test_idxs in cv.split(X, y, groups):
        vali_df = X.iloc[test_idxs]
        train_df = X.iloc[train_idxs]

        ##from shane
        query_tr = train_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
        X_train = train_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
        y_train = train_df["Label"]

        query_vali = vali_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
        valiq = vali_df['#QueryID']
        valid = vali_df['Docid']
        X_vali = vali_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
        y_vali = vali_df["Label"]

        query_test = test_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
        testq = test_df['#QueryID']
        testd = test_df['Docid']
        X_test = test_df.drop(['#QueryID', 'Docid'], axis=1)

        #scaler
        feature_scaler = MinMaxScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)

        #list of hyperparameter values to test
        param_grid = {'n_estimators': [10, 25, 50, 100, 150, 300],
                      'learning_rate': [0.5, 0.1, 0.2, 0.05, 0.01, 0.8],
                      'num_leaves': [24, 10, 5, 31, 50, 100],
                      'max_depth': [-1, 2, 6, 9, 31, 60]}

        #test each set of hyperparameters and save ngdc score in variable t
        for x in range(1296):
            marks = ParameterGrid(param_grid)[x]

            model = lightgbm.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                boosting_type='dart',
                n_estimators=marks.get('n_estimators'),
                learning_rate=marks.get('learning_rate'),
                num_leaves=marks.get('num_leaves'),
                max_depth=marks.get('max_depth')

            )

            model.fit(
                X=X_train,
                y=y_train,
                group=query_tr,
                eval_set=[(X_vali, y_vali)],
                eval_group=[query_vali],
                eval_metric='ndcg',
                eval_at=10,
                verbose=0
            )

            vali_score = model.predict(X_vali)

            fin_panda = pd.DataFrame(valiq)
            fin_panda = fin_panda.rename(columns={'#QueryID': 'QueryID'})

            fin_panda['Docid'] = valid
            fin_panda['Score'] = vali_score

            fin_panda.to_csv('fold.tsv', sep="\t", index=False, header=False)

            verbose = False
            this_os = platform.system()
            qrel_file = 'train.qrels'
            run_file = 'fold.tsv'
            # print ('OS = ',this_os)
            # print ('qrel file = ',qrel_file)
            # print ('system run file = ',run_file)
            rf = read_sort_run(run_file)
            output_file = 'current.run'
            with open(output_file, 'w') as f:
                f.write(rf)
            print(marks)
            out = run_trec(this_os, qrel_file, output_file, verbose)
            loc = out.find('0.')
            m = model.best_score_
            m = m.get('valid_0').get('ndcg@10')
            t[x][0] = t[x][0] + float(out[loc:loc + 6])
            print("model ndcg@10:  ", m)
            print("trec_eval ndcg: ", out[loc:loc + 6])
            if i == 0:
                t[x][1] = marks.get('n_estimators')
                t[x][2] = marks.get('learning_rate')
                t[x][3] = marks.get('num_leaves')
                t[x][4] = marks.get('max_depth')

        i = i + 1
    #get the average ndcg score for the folds by dividing by 10
    t[:, 0] = t[:, 0] / 10
    print("All ndcg:")
    print(t)
    #find highest average ndcg score
    best_idx = np.argmax(t[:, 0])
    parameters = t[best_idx, 1:5]
    print("Best parameters:")
    print(parameters)
    return parameters


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import lightgbm as lightgbm

    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import MinMaxScaler

    import subprocess
    import platform
    import os, sys

    #if sweep is TRUE - will set paramaters
    #if sweep if FALSE - will run kfold sweep
    sweep = True
    if sweep == True:

        # Set some parameters.
        # best parameters found from test run submission (S3799569-1.tsv)
        param_grid = {'n_estimators': [100],
                      'learning_rate': [0.2],
                      'num_leaves': [24],
                      'max_depth': [2]}

    else:
        # k fold search for best parameters
        parameters = parameter_sweep()
        param_grid = {'n_estimators': [int(parameters[0])],
                      'learning_rate': [parameters[1]],
                      'num_leaves': [int(parameters[2])],
                      'max_depth': [int(parameters[3])]}

    # get test and train data
    train_all = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')

    #group data for model and drop columns not needed
    query_tr = train_all.groupby('#QueryID')['#QueryID'].count().to_numpy()
    X_train = train_all.drop(['#QueryID', 'Label', 'Docid'], axis=1)
    y_train = train_all["Label"]

    query_test = test_df.groupby('#QueryID')['#QueryID'].count().to_numpy()
    testq = test_df['#QueryID']
    testd = test_df['Docid']
    X_test = test_df.drop(['#QueryID', 'Docid'], axis=1)

    #scaler
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    marks = ParameterGrid(param_grid)[0]

    model = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type='dart',
        n_estimators=marks.get('n_estimators'),
        learning_rate=marks.get('learning_rate'),
        num_leaves=marks.get('num_leaves'),
        max_depth=marks.get('max_depth')

    )

    model.fit(
        X=X_train,
        y=y_train,
        group=query_tr,
        verbose=0
    )

    #predict test data labels
    vali_score = model.predict(X_test)


    #combine queryid , docid and score into tsv file
    fin_panda = pd.DataFrame(testq)
    fin_panda = fin_panda.rename(columns={'#QueryID': 'QueryID'})

    fin_panda['Docid'] = testd
    fin_panda['Score'] = vali_score

    print(marks)

    fin_panda.to_csv('A2.run', sep="\t", index=False, header=False)

