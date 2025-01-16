import numpy as np
########### vgg16 ############
vgg_base_sup = {
    'Accuracy s.l.': 0.8283079060798548, 
    'F1 s.l.': 0.8908207334760737, 
    'ROC-AUC u.f.l.': 0.7810372841061803, 
    'Average Precision s.l.': 0.9152713409177777, 
    'Precision s.l.': 0.9312832316186938
}
vgg_abl_sup = {
    'Accuracy s.l.': 0.8249404491833031, 
    'F1 s.l.': 0.887861380559399, 
    'ROC-AUC u.f.l.': 0.7871441570241186, 
    'Average Precision s.l.': 0.9176754700224815, 
    'Precision s.l.': 0.9352615375365921
}
vgg_abl_ir_07_100 = {
    'Accuracy u.f.l.': 0.8254792422867514, 
    'F1 u.f.l.': 0.8886393897788287, 
    'ROC-AUC u.f.l.': 0.7821566944532685, 
    'Precision u.f.l.': 0.9322941522228617, 
    'Average Precision u.f.l.': 0.9156151966599668
}

########### effb3 ############
eff_abl_sup_not_scaled = {
    'Accuracy s.l.': 0.8222558864265929, 
    'F1 s.l.': 0.8860498055178415, 
    'ROC-AUC u.f.l.': 0.7811706640931675, 
    'Average Precision s.l.': 0.9156947618073822, 
    'Precision s.l.': 0.9333000252796648
}
eff_abl_sup_scale = {
    'Accuracy s.l.': 0.7981749192059095, 
    'F1 s.l.': 0.8651596040016539, 
    'ROC-AUC u.f.l.': 0.8100376463853592, 
    'Average Precision s.l.': 0.9273146644871455, 
    'Precision s.l.': 0.9553160401795602
}


def round_dict(dictionary):
    for key, val in dictionary.items():
        dictionary[key] = np.round(val, 2)
    return dictionary

print("vgg_base: ", round_dict(vgg_base_sup.copy()))
print("vgg_abl: ", round_dict(vgg_abl_sup.copy()))
print("vgg_abl_ir: ", round_dict(vgg_abl_ir.copy()))

print("eff_abl_not_scaled", round_dict(eff_abl_sup_not_scaled.copy()))
print("eff_abl_scale", round_dict(eff_abl_sup_scale.copy()))
