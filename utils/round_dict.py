import numpy as np


base = {
    'Accuracy s.l.': 0.8283079060798548, 
    'F1 s.l.': 0.8908207334760737, 
    'ROC-AUC u.f.l.': 0.7810372841061803, 
    'Average Precision s.l.': 0.9152713409177777, 
    'Precision s.l.': 0.9312832316186938
}
abl = {
    'Accuracy s.l.': 0.8249404491833031, 
    'F1 s.l.': 0.887861380559399, 
    'ROC-AUC u.f.l.': 0.7871441570241186, 
    'Average Precision s.l.': 0.9176754700224815, 
    'Precision s.l.': 0.9352615375365921
}

def round_dict(dictionary):
    for key, val in dictionary.items():
        dictionary[key] = np.round(val, 2)
    return dictionary

print("base: ", round_dict(base.copy()))
print("abl: ", round_dict(abl.copy()))