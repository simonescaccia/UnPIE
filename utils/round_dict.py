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

vgg_abl_ir_04_9229_100 = {'Accuracy u.f.l.': 0.8145686819419238, 'F1 u.f.l.': 0.8802766108984901, 'ROC-AUC u.f.l.': 0.7825404208888642, 'Average Precision u.f.l.': 0.9158939841649802, 'Precision u.f.l.': 0.9348654157978049}
vgg_abl_ir_04_4096_100 = {'Accuracy u.f.l.': 0.8260180353901997, 'F1 u.f.l.': 0.8889093768488946, 'ROC-AUC u.f.l.': 0.784107829386639, 'Average Precision u.f.l.': 0.9164187413844654, 'Precision u.f.l.': 0.9332893703600236}
vgg_abl_ir_04_2048_100 = {'Accuracy u.f.l.': 0.8092629877495463, 'F1 u.f.l.': 0.8755067728206205, 'ROC-AUC u.f.l.': 0.7925669981204352, 'Average Precision u.f.l.': 0.9200078051974101, 'Precision u.f.l.': 0.941908851693255}
vgg_abl_ir_04_1024_100 = {'Accuracy u.f.l.': 0.8159979015426498, 'F1 u.f.l.': 0.8813615440703619, 'ROC-AUC u.f.l.': 0.7845897000480282, 'Average Precision u.f.l.': 0.9167618880119226, 'Precision u.f.l.': 0.935801854766117}
vgg_abl_ir_04_512_100 = {'Accuracy u.f.l.': 0.8244016560798548, 'F1 u.f.l.': 0.8878895536807023, 'ROC-AUC u.f.l.': 0.7813824185205107, 'Average Precision u.f.l.': 0.9154051140716, 'Precision u.f.l.': 0.9322297118869538}
vgg_abl_ir_04_256_100 = {'Accuracy u.f.l.': 0.8079684664246825, 'F1 u.f.l.': 0.8739542753409928, 'ROC-AUC u.f.l.': 0.8008067172025928, 'Average Precision u.f.l.': 0.923149626792096, 'Precision u.f.l.': 0.9465698317036302}
vgg_abl_ir_04_128_100 = {'Accuracy u.f.l.': 0.8003729015426498, 'F1 u.f.l.': 0.8676261166249911, 'ROC-AUC u.f.l.': 0.8050900172726863, 'Average Precision u.f.l.': 0.9253278966871115, 'Precision u.f.l.': 0.9516182160892606}
vgg_abl_ir_04_64_100 = {'Accuracy u.f.l.': 0.8254792422867514, 'F1 u.f.l.': 0.8885850533761059, 'ROC-AUC u.f.l.': 0.7829478126647856, 'Average Precision u.f.l.': 0.915972103376972, 'Precision u.f.l.': 0.9328043625656387}
vgg_abl_ir_04_32_100 = {'Accuracy u.f.l.': 0.789058246370236, 'F1 u.f.l.': 0.8585457370777096, 'ROC-AUC u.f.l.': 0.8040497221892562, 'Average Precision u.f.l.': 0.9250575693907103, 'Precision u.f.l.': 0.95404549734173}
vgg_abl_ir_04_16_100 = {'Accuracy u.f.l.': 0.821614110707804, 'F1 u.f.l.': 0.8857614873374362, 'ROC-AUC u.f.l.': 0.7817164162605587, 'Average Precision u.f.l.': 0.9156217778219342, 'Precision u.f.l.': 0.9331056162541066}
vgg_abl_ir_04_1_100 = {'Accuracy u.f.l.': 0.8087766560798548, 'F1 u.f.l.': 0.8747273734093771, 'ROC-AUC u.f.l.': 0.7981911488915648, 'Average Precision u.f.l.': 0.9222554437989998, 'Precision u.f.l.': 0.9451308309419268}
vgg_abl_ir_04_0_100 = {'Accuracy u.f.l.': 0.8016674228675136, 'F1 u.f.l.': 0.868375401282314, 'ROC-AUC u.f.l.': 0.8077406018333749, 'Average Precision u.f.l.': 0.9261048839329634, 'Precision u.f.l.': 0.9525829647753293}

vgg_abl_ir_04_0_50 = {'Accuracy u.f.l.': 0.8171279491833031, 'F1 u.f.l.': 0.88245768574466, 'ROC-AUC u.f.l.': 0.7802857509863125, 'Average Precision u.f.l.': 0.9151731924606139, 'Precision u.f.l.': 0.9333108581869528}
vgg_abl_ir_04_0_75 = {'Accuracy u.f.l.': 0.8083200998185118, 'F1 u.f.l.': 0.8750407136981879, 'ROC-AUC u.f.l.': 0.7886342539736164, 'Average Precision u.f.l.': 0.9183840854906019, 'Precision u.f.l.': 0.9397889958213596}
vgg_abl_ir_04_0_100 = {'Accuracy u.f.l.': 0.8016674228675136, 'F1 u.f.l.': 0.868375401282314, 'ROC-AUC u.f.l.': 0.8077406018333749, 'Average Precision u.f.l.': 0.9261048839329634, 'Precision u.f.l.': 0.9525829647753293}
vgg_abl_ir_04_0_125 = {'Accuracy u.f.l.': 0.824805750907441, 'F1 u.f.l.': 0.8873242693104907, 'ROC-AUC u.f.l.': 0.7947209594351178, 'Average Precision u.f.l.': 0.9204392505614425, 'Precision u.f.l.': 0.9390113369446257}
vgg_abl_ir_04_0_150 = {'Accuracy u.f.l.': 0.7990259187840291, 'F1 u.f.l.': 0.869323762566076, 'ROC-AUC u.f.l.': 0.7699065508853483, 'Average Precision u.f.l.': 0.9115870655808141, 'Precision u.f.l.': 0.9319666827487377}
vgg_abl_ir_04_0_200 = {'Accuracy u.f.l.': 0.804683246370236, 'F1 u.f.l.': 0.8714683889960683, 'ROC-AUC u.f.l.': 0.8000630762281645, 'Average Precision u.f.l.': 0.9233118565292275, 'Precision u.f.l.': 0.9476554139594223}
vgg_abl_ir_04_0_250 = {'Accuracy u.f.l.': 0.8212100158802178, 'F1 u.f.l.': 0.8847481886399101, 'ROC-AUC u.f.l.': 0.7918769379360706, 'Average Precision u.f.l.': 0.9192894898859006, 'Precision u.f.l.': 0.9382604058077868}
vgg_abl_ir_04_0_275 = {'Accuracy u.f.l.': 0.8213035957350272, 'F1 u.f.l.': 0.885558843640464, 'ROC-AUC u.f.l.': 0.7809309521629785, 'Average Precision u.f.l.': 0.9153053666738759, 'Precision u.f.l.': 0.9327115938746866}
vgg_abl_ir_04_0_300 = {'Accuracy u.f.l.': 0.8162672980943739, 'F1 u.f.l.': 0.8813991885677717, 'ROC-AUC u.f.l.': 0.7863958081754469, 'Average Precision u.f.l.': 0.9177910586468485, 'Precision u.f.l.': 0.9372785098489607}
vgg_abl_ir_04_0_325 = {'Accuracy u.f.l.': 0.8089000113430127, 'F1 u.f.l.': 0.8762217023290367, 'ROC-AUC u.f.l.': 0.7788206397285073, 'Average Precision u.f.l.': 0.9147247297259802, 'Precision u.f.l.': 0.934604349309171}
vgg_abl_ir_04_0_350 = {'Accuracy u.f.l.': 0.8164544578039927, 'F1 u.f.l.': 0.8820702572769619, 'ROC-AUC u.f.l.': 0.7792858855415673, 'Average Precision u.f.l.': 0.9148918718143652, 'Precision u.f.l.': 0.9330330726063246}
vgg_abl_ir_04_0_400 = {'Accuracy u.f.l.': 0.7917933303085299, 'F1 u.f.l.': 0.8624168407715408, 'ROC-AUC u.f.l.': 0.7845815183343404, 'Average Precision u.f.l.': 0.9174884152105081, 'Precision u.f.l.': 0.942118850484719}
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

print("vgg_abl_ir_04_9229_100: ", round_dict(vgg_abl_ir_04_9229_100.copy()))
print("vgg_abl_ir_04_4096_100: ", round_dict(vgg_abl_ir_04_4096_100.copy()))
print("vgg_abl_ir_04_2048_100: ", round_dict(vgg_abl_ir_04_2048_100.copy()))
print("vgg_abl_ir_04_1024_100: ", round_dict(vgg_abl_ir_04_1024_100.copy()))
print("vgg_abl_ir_04_512_100: ", round_dict(vgg_abl_ir_04_512_100.copy()))
print("vgg_abl_ir_04_256_100: ", round_dict(vgg_abl_ir_04_256_100.copy()))
print("vgg_abl_ir_04_128_100: ", round_dict(vgg_abl_ir_04_128_100.copy()))
print("vgg_abl_ir_04_64_100: ", round_dict(vgg_abl_ir_04_64_100.copy()))
print("vgg_abl_ir_04_32_100: ", round_dict(vgg_abl_ir_04_32_100.copy()))
print("vgg_abl_ir_04_16_100: ", round_dict(vgg_abl_ir_04_16_100.copy()))
print("vgg_abl_ir_04_1_100: ", round_dict(vgg_abl_ir_04_1_100.copy()))
print("vgg_abl_ir_04_0_100: ", round_dict(vgg_abl_ir_04_0_100.copy()))


print("vgg_abl_ir_04_0_50: ", round_dict(vgg_abl_ir_04_0_50.copy()))
print("vgg_abl_ir_04_0_75: ", round_dict(vgg_abl_ir_04_0_75.copy()))
print("vgg_abl_ir_04_0_100: ", round_dict(vgg_abl_ir_04_0_100.copy()))
print("vgg_abl_ir_04_0_125: ", round_dict(vgg_abl_ir_04_0_125.copy()))
print("vgg_abl_ir_04_0_150: ", round_dict(vgg_abl_ir_04_0_150.copy()))
print("vgg_abl_ir_04_0_200: ", round_dict(vgg_abl_ir_04_0_200.copy()))
print("vgg_abl_ir_04_0_250: ", round_dict(vgg_abl_ir_04_0_250.copy()))
print("vgg_abl_ir_04_0_275: ", round_dict(vgg_abl_ir_04_0_275.copy()))
print("vgg_abl_ir_04_0_300: ", round_dict(vgg_abl_ir_04_0_300.copy()))
print("vgg_abl_ir_04_0_325: ", round_dict(vgg_abl_ir_04_0_325.copy()))
print("vgg_abl_ir_04_0_350: ", round_dict(vgg_abl_ir_04_0_350.copy()))
print("vgg_abl_ir_04_0_400: ", round_dict(vgg_abl_ir_04_0_400.copy()))

print("eff_abl_not_scaled", round_dict(eff_abl_sup_not_scaled.copy()))
print("eff_abl_scale", round_dict(eff_abl_sup_scale.copy()))
