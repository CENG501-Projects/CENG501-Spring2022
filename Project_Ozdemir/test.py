from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from utils import calc_accuracy



def classify_nodes(C, train_emb, train_lbs, test_emb, test_lbs):
    clf = svm.SVC(C=C)
    clf.fit(train_emb, train_lbs)

    preds = clf.predict(test_emb)
    acc = calc_accuracy(preds, test_lbs)

    return preds, acc


def cluster_nodes(num_clusters, random_state, train_emb, test_emb, test_lbs):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(train_emb)

    preds = kmeans.predict(test_emb)

    NMI_score = normalized_mutual_info_score(test_lbs, preds)
    ARI_score = adjusted_rand_score(test_lbs, preds)

    return preds, NMI_score, ARI_score