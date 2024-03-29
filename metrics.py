import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def get_metric_evaluations(evaluated_model,
                            X_train,
                            y_true,
                            model_config_name, 
                            experiment_config_name, 
                            description='', ):
    y_pred = evaluated_model.predict(X_train)
    y_score = evaluated_model.predict_proba(X_train)[:,1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {'Description': description,
                        'Precision': precision_score(y_true, y_pred),
                        'Recal': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'Experiment config':experiment_config_name,
                        'Model config': model_config_name
                        }
    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results)
