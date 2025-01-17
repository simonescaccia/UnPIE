import numpy as np
########### vgg16 ############
vgg_base_sup = {'Accuracy s.l.': 0.8283079060798548, 'F1 s.l.': 0.8908207334760737, 'ROC-AUC u.f.l.': 0.7810372841061803, 'Average Precision s.l.': 0.9152713409177777, 'Precision s.l.': 0.9312832316186938}
vgg_abl_sup = {'Accuracy s.l.': 0.8249404491833031, 'F1 s.l.': 0.887861380559399, 'ROC-AUC u.f.l.': 0.7871441570241186, 'Average Precision s.l.': 0.9176754700224815, 'Precision s.l.': 0.9352615375365921}
vgg_abl_ir_13_0_100 = {'Accuracy u.f.l.': 0.8184749319419238, 'F1 u.f.l.': 0.8836520032933399, 'ROC-AUC u.f.l.': 0.7783547978828325, 'Average Precision u.f.l.': 0.9144426189805774, 'Precision u.f.l.': 0.9320185102279288}
vgg_abl_ir_12_0_100 = {'Accuracy u.f.l.': 0.8069731170598912, 'F1 u.f.l.': 0.8743492337441519, 'ROC-AUC u.f.l.': 0.7842599170093796, 'Average Precision u.f.l.': 0.9167461994933196, 'Precision u.f.l.': 0.9376921313757927}
vgg_abl_ir_11_0_100 = {'Accuracy u.f.l.': 0.824805750907441, 'F1 u.f.l.': 0.8880868658758246, 'ROC-AUC u.f.l.': 0.7828244978296235, 'Average Precision u.f.l.': 0.915947015944829, 'Precision u.f.l.': 0.9329143482677816}
vgg_abl_ir_10_0_100 = {'Accuracy u.f.l.': 0.8229199750453721, 'F1 u.f.l.': 0.8868427310969106, 'ROC-AUC u.f.l.': 0.7805205837705655, 'Average Precision u.f.l.': 0.9151047151059883, 'Precision u.f.l.': 0.9321261800700309}
vgg_abl_ir_09_0_100 = {'Accuracy u.f.l.': 0.8182055353901997, 'F1 u.f.l.': 0.8828898657382078, 'ROC-AUC u.f.l.': 0.7851599777748292, 'Average Precision u.f.l.': 0.9169529708497524, 'Precision u.f.l.': 0.9355863783236559}
vgg_abl_ir_08_0_100 = {'Accuracy u.f.l.': 0.7975031193284937, 'F1 u.f.l.': 0.867381870081026, 'ROC-AUC u.f.l.': 0.7794335665066969, 'Average Precision u.f.l.': 0.91554301766674, 'Precision u.f.l.': 0.9378989528812758}
vgg_abl_ir_07_0_100 = {'Accuracy u.f.l.': 0.8041969147005446, 'F1 u.f.l.': 0.8719646430624637, 'ROC-AUC u.f.l.': 0.7869983572924684, 'Average Precision u.f.l.': 0.918058311503254, 'Precision u.f.l.': 0.9400467772662523}
vgg_abl_ir_06_0_100 = {'Accuracy u.f.l.': 0.8057494895644283, 'F1 u.f.l.': 0.8729540357533164, 'ROC-AUC u.f.l.': 0.7919511781881873, 'Average Precision u.f.l.': 0.9197700403376238, 'Precision u.f.l.': 0.9422616608649217 }
vgg_abl_ir_05_0_100 = {'Accuracy u.f.l.': 0.8208059210526316, 'F1 u.f.l.': 0.8853158493458622, 'ROC-AUC u.f.l.': 0.7800308915504532, 'Average Precision u.f.l.': 0.9150291669339978, 'Precision u.f.l.': 0.932412652041798}
vgg_abl_ir_04_0_100 = {'Accuracy u.f.l.': 0.8016674228675136, 'F1 u.f.l.': 0.868375401282314, 'ROC-AUC u.f.l.': 0.8077406018333749, 'Average Precision u.f.l.': 0.9261048839329634, 'Precision u.f.l.': 0.9525829647753293}
vgg_abl_ir_03_0_100 = {'Accuracy u.f.l.': 0.8052220394736843, 'F1 u.f.l.': 0.8731443276875087, 'ROC-AUC u.f.l.': 0.7828889121389415, 'Average Precision u.f.l.': 0.9163135997949909, 'Precision u.f.l.': 0.9374179020772785}
vgg_abl_ir_02_0_100 = {'Accuracy u.f.l.': 0.8178836774047188, 'F1 u.f.l.': 0.8825141259333621, 'ROC-AUC u.f.l.': 0.7875814833728094, 'Average Precision u.f.l.': 0.9180613606728178, 'Precision u.f.l.': 0.9373312316857525}
########### effb3 ############
eff_abl_sup_not_scaled = {'Accuracy s.l.': 0.8222558864265929, 'F1 s.l.': 0.8860498055178415, 'ROC-AUC u.f.l.': 0.7811706640931675, 'Average Precision s.l.': 0.9156947618073822, 'Precision s.l.': 0.9333000252796648}
eff_abl_sup_scale = {'Accuracy s.l.': 0.7981749192059095, 'F1 s.l.': 0.8651596040016539, 'ROC-AUC u.f.l.': 0.8100376463853592, 'Average Precision s.l.': 0.9273146644871455, 'Precision s.l.': 0.9553160401795602}


def round_dict(dictionary):
    for key, val in dictionary.items():
        dictionary[key] = np.round(val, 2)
    return dictionary

print("vgg_base: ", round_dict(vgg_base_sup.copy()))
print("vgg_abl: ", round_dict(vgg_abl_sup.copy()))
print("vgg_abl_ir_13_0_100: ", round_dict(vgg_abl_ir_13_0_100.copy()))
print("vgg_abl_ir_12_0_100: ", round_dict(vgg_abl_ir_12_0_100.copy()))
print("vgg_abl_ir_11_0_100: ", round_dict(vgg_abl_ir_11_0_100.copy()))
print("vgg_abl_ir_10_0_100: ", round_dict(vgg_abl_ir_10_0_100.copy()))
print("vgg_abl_ir_09_0_100: ", round_dict(vgg_abl_ir_09_0_100.copy()))
print("vgg_abl_ir_08_0_100: ", round_dict(vgg_abl_ir_08_0_100.copy()))
print("vgg_abl_ir_07_0_100: ", round_dict(vgg_abl_ir_07_0_100.copy()))
print("vgg_abl_ir_06_0_100: ", round_dict(vgg_abl_ir_06_0_100.copy()))
print("vgg_abl_ir_05_0_100: ", round_dict(vgg_abl_ir_05_0_100.copy()))
print("vgg_abl_ir_04_0_100: ", round_dict(vgg_abl_ir_04_0_100.copy()))
print("vgg_abl_ir_03_0_100: ", round_dict(vgg_abl_ir_03_0_100.copy()))
print("vgg_abl_ir_02_0_100: ", round_dict(vgg_abl_ir_02_0_100.copy()))

print("eff_abl_not_scaled", round_dict(eff_abl_sup_not_scaled.copy()))
print("eff_abl_scale", round_dict(eff_abl_sup_scale.copy()))
