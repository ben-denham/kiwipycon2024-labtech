import labtech as lt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def experiment(max_leaves: int):
    txt_train = fetch_20newsgroups(subset='train')
    txt_test = fetch_20newsgroups(subset='test')

    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(txt_train.data)
    X_test = vectorizer.transform(txt_test.data)

    classifier = RandomForestClassifier(max_leaf_nodes=max_leaves)
    classifier.fit(X_train, txt_train.target)
    y_pred = classifier.predict(X_test)

    return accuracy_score(txt_test.target, y_pred)


@lt.task
class Experiment:
    max_leaves: int
 
    def run(self):
        return experiment(max_leaves=self.max_leaves)


if __name__ == '__main__':
    experiments = [Experiment(max_leaves=max_leaves) for max_leaves in [10, 50, 90]]
     
    lab = lt.Lab(storage=None, max_workers=3)
    results = lab.run_tasks(experiments)
    print(results)