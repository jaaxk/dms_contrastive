from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score


def ridge_metrics(train_projections, train_quartiles, test_projections, test_quartiles):
    ridge = RidgeClassifier()
    test_quartiles = [1 if q == 'high' else 0 for q in test_quartiles]
    train_quartiles = [1 if q == 'high' else 0 for q in train_quartiles]

    ridge.fit(train_projections, train_quartiles)

    predictions = ridge.predict(test_projections)
    accuracy = accuracy_score(test_quartiles, predictions)
    precision = precision_score(test_quartiles, predictions)
    recall = recall_score(test_quartiles, predictions)
    f1 = f1_score(test_quartiles, predictions)

    # Compute AUC
    probs = ridge.decision_function(test_projections)
    auc = roc_auc_score(test_quartiles, probs)
    
    return accuracy, precision, recall, f1, auc

def knn_metrics(test_embeddings, test_quartiles, train_embeddings, train_quartiles):
    if len(train_embeddings) < 5:
        #print("WARNING: Not enough data to train KNN, returning None for all metrics.")
        return None, None, None, None, None  # Not enough data to train KNN
    neigh = KNeighborsClassifier(n_neighbors=5)
    test_quartiles = [1 if q == 'high' else 0 for q in test_quartiles]
    train_quartiles = [1 if q == 'high' else 0 for q in train_quartiles]
    neigh.fit(train_embeddings, train_quartiles)
    
    predictions = neigh.predict(test_embeddings)
    accuracy = accuracy_score(test_quartiles, predictions)
    precision = precision_score(test_quartiles, predictions)
    recall = recall_score(test_quartiles, predictions)
    f1 = f1_score(test_quartiles, predictions)

    probs = neigh.predict_proba(test_embeddings)
    auc = roc_auc_score(test_quartiles, probs[:, 1])  # Use probabilities for the positive class
    
    return accuracy, precision, recall, f1, auc

def contrastive_metrics(similarities, labels):
    binary_preds = [1 if sim > 0.5 else 0 for sim in similarities]
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    return accuracy, precision, recall, f1

def kmeans_metrics(projections, quartiles, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(projections)
    kmeans_labels = kmeans.labels_

    quartiles = [1 if q == 'high' else 0 for q in quartiles]

    accuracy = max(accuracy_score(quartiles, kmeans_labels), 1 - accuracy_score(quartiles, kmeans_labels))
    precision = max(precision_score(quartiles, kmeans_labels), 1 - precision_score(quartiles, kmeans_labels))
    recall = max(recall_score(quartiles, kmeans_labels), 1 - recall_score(quartiles, kmeans_labels))
    f1 = max(f1_score(quartiles, kmeans_labels), 1 - f1_score(quartiles, kmeans_labels))

    return accuracy, precision, recall, f1

def logreg_metrics(train_projections, train_quartiles, test_projections, test_quartiles, sample_train=1000):
    # Sample data for efficiency
    if len(train_projections) > sample_train:
        train_indices = np.random.choice(len(train_projections), sample_train, replace=False)
        train_proj_sample = [train_projections[i] for i in train_indices]
        train_quartile_sample = [train_quartiles[i] for i in train_indices]
    else:
        train_proj_sample = train_projections
        train_quartile_sample = train_quartiles
    
    # Convert to numpy arrays
    X_train = np.array(train_proj_sample)
    y_train = np.array([1 if q == 'high' else 0 for q in train_quartile_sample])
    
    X_test = np.array(test_projections)
    y_test = np.array([1 if q == 'high' else 0 for q in test_quartiles])
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1