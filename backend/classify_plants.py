from fastai.vision.all import *
from fastai.vision.utils import *
import matplotlib.pyplot as plt

learn2 = load_learner('export.pkl')

categories = ('Aptenia cordifolia', 'aloe vera', 'epipremnum aureum', 'sansevieria trifasciata', 'spathiphyllum')

def classify_image(img, threshold=0.7):
    imgToPredict = plt.imread(img)
    pred = learn2.predict(imgToPredict)
    pred_probs_dec = [f"{p:.20f}" for p in pred[2].tolist()]
    max_prob = max(pred[2].tolist())
    if max_prob >= threshold:
        return pred[0], dict(zip(categories, pred_probs_dec)), max_prob
    else:
        return "Uncertain", dict(zip(categories, pred_probs_dec)), max_prob

def getClassesForLearner(learner):
    return learner.dls.vocab

def getLearner():
    return learn2

def getCategories():
    return categories

def getProbsInDecimal(pred):
    pred_probs_dec = [f"{p:.20f}" for p in pred[2].tolist()]
    return pred_probs_dec