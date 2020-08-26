"""
KDDCUP2020 MLTrack2
https://www.biendata.xyz/competition/kddcup_2020/

Author: NTT DOCOMO LABS
License: MIT
"""

import lightgbm as lgb

from models import GCN, SAGE

class BaseClassifier():
    def __init__(self, ):
        hogehoge
        pass
    def __del__(self):
        hogehoge
        pass

class RobustClassifier():
    def __init__(self, ):
        self.model1="GCN"
        self.model2="SAGE"
        self.model3="LightGBM"
        pass
    def train():
        pass
    def ensenmble():
        pass
        
    def predict(self, modelname):
        if modelname=="GCN":
            pass
        elif modelname=="SAGE":
            pass
        pass
    def __del__(self):
        hogehoge
        pass

class LGBClassifier():
    def __init__(self, data, params):
        """LightGBMでラベルを予測する"""
        X = data.x
        
    def train(self, X_train, y_train,):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        model = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round=5000,
                          early_stopping_rounds=50,
                          evals_result=score,
                          feval=accuracy,
                          verbose_eval=50
                         )
        return model
    
    def test(self, model):
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        label_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def save_model(self):
        with open(path, "wb") as f:
            pickle.dump(model, f)
            
    def load_model(self):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


def accuracy(preds, train_data):
    """LightGBMに渡す評価関数"""
    y_test = train_data.get_label()
    y_preds = np.argmax(preds.reshape(18, len(preds)//18), axis=0)
    eval_name = "accuracy"
    eval_result = accuracy_score(y_test, y_preds)
    is_higher_better = True
    return eval_name, eval_result, is_higher_better