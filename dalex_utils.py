import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



def calculate_regression_measures_all(y, y_hat, protected, privileged):

    unique_protected = np.unique(protected)

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])

    y_u = ((y - y.mean()) / y.std()).reshape(-1, 1)
    s_u = ((y_hat - y_hat.mean()) / y_hat.std()).reshape(-1, 1)

    a = np.where(protected == privileged, 1, 0)

    p_s = LogisticRegression(class_weight='balanced')
    p_ys = LogisticRegression(class_weight='balanced')
    p_y = LogisticRegression(class_weight='balanced')

    p_s.fit(s_u, a)
    p_y.fit(y_u, a)
    p_ys.fit(np.c_[y_u, s_u], a)

    pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
    pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
    pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]

    n = len(a)

    r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
    r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
    r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

    data = {'subgroup': privileged,
            'independence': r_ind,
            'separation': r_sep,
            'sufficiency': r_suf}


    return data



def dalex_results(df, protected, target_feature="target", prediction_col="prediction"):
    print(f"Dalex Results for {protected}")
    y_true = df[target_feature].values
    y_pred = df['prediction'].values
    unique_protected = df[protected].unique()
    result_list = []
    for privileged in unique_protected:
        result = calculate_regression_measures_all(y_true, y_pred, df[protected], privileged)
        result_list.append(result)
    result_df = pd.DataFrame(result_list).sort_values("subgroup")
    return result_df