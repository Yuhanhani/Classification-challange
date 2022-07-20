#import scikit-learn
import skll
#import sklearn



def quadratic_weighted_kappa(y_true, y_pred):
    kappa_score = skll.kappa(y_true, y_pred, weights='quadratic', allow_off_by_one=False)
