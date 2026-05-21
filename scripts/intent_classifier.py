"""
Week 7: Buyer Intent Classification

IntentClassifier uses TF-IDF + Logistic Regression to classify real estate
queries into: browsing, researching, ready_to_buy.

IntentQueryParser integrates IntentClassifier with Week 4 QueryParser
for richer combined output.
"""

import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class IntentClassifier:
    """Multi-class intent classifier for real estate buyer queries."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.labels = ['browsing', 'ready_to_buy', 'researching']
        self._is_trained = False

    def train(self, queries, labels):
        """Fit vectorizer and classifier on labeled queries.

        Args:
            queries: list of query strings
            labels: list of intent labels
        """
        X = self.vectorizer.fit_transform(queries)
        self.model.fit(X, labels)
        self.labels = sorted(list(set(labels)))
        self._is_trained = True

    def predict(self, query):
        """Predict intent and confidence for a single query.

        Args:
            query: query string

        Returns:
            tuple: (intent_label, confidence_score)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([query])
        probas = self.model.predict_proba(X)[0]
        intent = self.model.classes_[probas.argmax()]
        confidence = float(probas.max())
        return intent, confidence

    def predict_batch(self, queries):
        """Predict intents for multiple queries.

        Args:
            queries: list of query strings

        Returns:
            list of (intent, confidence) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform(queries)
        probas = self.model.predict_proba(X)
        results = []
        for p in probas:
            intent = self.model.classes_[p.argmax()]
            confidence = float(p.max())
            results.append((intent, confidence))
        return results

    def predict_proba(self, query):
        """Get full probability distribution for a query.

        Returns:
            dict mapping intent label to probability
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([query])
        probas = self.model.predict_proba(X)[0]
        return dict(zip(self.model.classes_, [float(p) for p in probas]))

    def evaluate(self, test_queries, test_labels):
        """Evaluate model on test set.

        Returns:
            dict with accuracy, classification_report (str), confusion_matrix
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform(test_queries)
        preds = self.model.predict(X)
        acc = accuracy_score(test_labels, preds)
        report = classification_report(test_labels, preds)
        cm = confusion_matrix(test_labels, preds, labels=self.labels)
        return {
            'accuracy': float(acc),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': preds.tolist(),
        }

    def save_model(self, path):
        """Persist trained model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'labels': self.labels,
            }, f)

    def load_model(self, path):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        self.labels = data['labels']
        self._is_trained = True


class IntentQueryParser:
    """Combines IntentClassifier with Week 4 QueryParser for richer output."""

    def __init__(self, classifier, query_parser):
        """
        Args:
            classifier: trained IntentClassifier instance
            query_parser: QueryParser instance from Week 4
        """
        self.classifier = classifier
        self.parser = query_parser

    def analyze(self, query):
        """Analyze a query for both intent and structured filters.

        Returns:
            dict with intent, confidence, probabilities, filters, sql, params
        """
        intent, confidence = self.classifier.predict(query)
        probas = self.classifier.predict_proba(query)
        filters = self.parser.parse(query)
        sql, params = self.parser.to_sql(filters)

        return {
            'query': query,
            'intent': intent,
            'confidence': round(confidence, 4),
            'probabilities': {k: round(v, 4) for k, v in probas.items()},
            'filters': filters,
            'sql': sql,
            'params': params,
        }


if __name__ == '__main__':
    # Quick demo
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base, 'data', 'processed', 'intent_dataset.json')

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    queries = [d['query'] for d in dataset]
    labels = [d['intent'] for d in dataset]

    clf = IntentClassifier()
    clf.train(queries, labels)

    test_queries = [
        "What's available in Portland?",
        "Compare condos vs townhouses in Irvine",
        "3 bed 2 bath under $500k in Portland with pool",
    ]
    for q in test_queries:
        intent, conf = clf.predict(q)
        print(f"  [{intent:>12}] ({conf:.2f}) {q}")
