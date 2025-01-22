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

vgg_abl_la_4_512_5 = {'Accuracy u.f.l.': 0.7978958711433757, 'F1 u.f.l.': 0.8663308629851819, 'ROC-AUC u.f.l.': 0.7968018287976012, 'Average Precision u.f.l.': 0.9220586403739389, 'Precision u.f.l.': 0.9474114851117553}
vgg_abl_la_8_512_5 = {'Accuracy u.f.l.': 0.8262874319419238, 'F1 u.f.l.': 0.8888184802968615, 'ROC-AUC u.f.l.': 0.7883978620803648, 'Average Precision u.f.l.': 0.9180815853734209, 'Precision u.f.l.': 0.9355662074922388}
vgg_abl_la_16_512_5 = {'Accuracy u.f.l.': 0.8184224705081671, 'F1 u.f.l.': 0.8822062040840988, 'ROC-AUC u.f.l.': 0.7976794359549061, 'Average Precision u.f.l.': 0.9216681744388127, 'Precision u.f.l.': 0.9422148038919111}
vgg_abl_la_32_512_5 = {'Accuracy u.f.l.': 0.8093154491833031, 'F1 u.f.l.': 0.876747191697416, 'ROC-AUC u.f.l.': 0.7769941560405094, 'Average Precision u.f.l.': 0.914101169881236, 'Precision u.f.l.': 0.9333638029950063}
vgg_abl_la_64_512_5 = {'Accuracy u.f.l.': 0.818287772232305, 'F1 u.f.l.': 0.8830404023910364, 'ROC-AUC u.f.l.': 0.7848044669692515, 'Average Precision u.f.l.': 0.9168717930788349, 'Precision u.f.l.': 0.9354914857673945}
vgg_abl_la_128_512_5 = {'Accuracy u.f.l.': 0.818287772232305, 'F1 u.f.l.': 0.8830404023910364, 'ROC-AUC u.f.l.': 0.7848044669692515, 'Average Precision u.f.l.': 0.9168717930788349, 'Precision u.f.l.': 0.9354914857673945}

vgg_abl_la_2_16_5_warm = {'Accuracy u.f.l.': 0.8257486388384755, 'F1 u.f.l.': 0.8886956692425811, 'ROC-AUC u.f.l.': 0.7843406602253261, 'Average Precision u.f.l.': 0.9164821924060995, 'Precision u.f.l.': 0.9334479440606824}
vgg_abl_la_2_32_5_warm = {'Accuracy u.f.l.': 0.8205365245009075, 'F1 u.f.l.': 0.8839079743040601, 'ROC-AUC u.f.l.': 0.7968711255421593, 'Average Precision u.f.l.': 0.9213421418339642, 'Precision u.f.l.': 0.9413606728260172}
vgg_abl_la_2_64_5_warm = {'Accuracy u.f.l.': 0.8211164360254084, 'F1 u.f.l.': 0.8849448741162871, 'ROC-AUC u.f.l.': 0.7884079086947788, 'Average Precision u.f.l.': 0.918148522970787, 'Precision u.f.l.': 0.93675032999403}
vgg_abl_la_2_128_5_warm = {'Accuracy u.f.l.': 0.8239039813974592, 'F1 u.f.l.': 0.8869098353662833, 'ROC-AUC u.f.l.': 0.790118588274357, 'Average Precision u.f.l.': 0.9188695178036024, 'Precision u.f.l.': 0.9371600390566251}
vgg_abl_la_2_256_5_warm = {'Accuracy u.f.l.': 0.8252098457350272, 'F1 u.f.l.': 0.8884050237114332, 'ROC-AUC u.f.l.': 0.7830765777590813, 'Average Precision u.f.l.': 0.916046482037964, 'Precision u.f.l.': 0.9329289199137449}
vgg_abl_la_2_512_5_warm = {'Accuracy u.f.l.': 0.7970579060798548, 'F1 u.f.l.': 0.8644222977237022, 'ROC-AUC u.f.l.': 0.8104951675856711, 'Average Precision u.f.l.': 0.927330299725275, 'Precision u.f.l.': 0.955443516742688}
vgg_abl_la_2_1024_5_warm = {'Accuracy u.f.l.': 0.8189201451905627, 'F1 u.f.l.': 0.8832036134909237, 'ROC-AUC u.f.l.': 0.7889186342850528, 'Average Precision u.f.l.': 0.9183380806603446, 'Precision u.f.l.': 0.937416930269469}

vgg_abl_bal_sup = {'Accuracy s.l.': 0.7980587121212122, 'F1 s.l.': 0.7891993612278217, 'ROC-AUC u.f.l.': 0.8011841610098768, 'Average Precision s.l.': 0.7379875199151269, 'Precision s.l.': 0.8085847498524893}
vgg_abl_bal_ir_noise = {'Accuracy u.f.l.': 0.7813683712121212, 'F1 u.f.l.': 0.7568803002012345, 'ROC-AUC u.f.l.': 0.7808197835612741, 'Average Precision u.f.l.': 0.7272578806485946, 'Precision u.f.l.': 0.8290717130931632}
vgg_abl_bal_ir_no_noise = {'Accuracy u.f.l.': 0.7994791666666667, 'F1 u.f.l.': 0.7868197681912757, 'ROC-AUC u.f.l.': 0.8004515397597324, 'Average Precision u.f.l.': 0.7413708404897282, 'Precision u.f.l.': 0.8204412323386671}
vgg_abl_bal_la_bi_group = {'Accuracy u.f.l.': 0.8053977272727273, 'F1 u.f.l.': 0.7941036941758377, 'ROC-AUC u.f.l.': 0.8059748207381479, 'Average Precision u.f.l.': 0.7479002628845476, 'Precision u.f.l.': 0.8244058887100801}
vgg_abl_bal_la_multi_group = {'Accuracy u.f.l.': 0.7948626893939394, 'F1 u.f.l.': 0.7856243349701605, 'ROC-AUC u.f.l.': 0.7960742240776848, 'Average Precision u.f.l.': 0.7324866101316089, 'Precision u.f.l.': 0.8035911033504243}

vgg_abl_la_2_512_1_warm = {'Accuracy u.f.l.': 0.8256139405626135, 'F1 u.f.l.': 0.8888477617605964, 'ROC-AUC u.f.l.': 0.7807876115801029, 'Average Precision u.f.l.': 0.9151064182785198, 'Precision u.f.l.': 0.9315614030564242}
vgg_abl_la_2_512_2_warm = {'Accuracy u.f.l.': 0.818744328493648, 'F1 u.f.l.': 0.8825702716137477, 'ROC-AUC u.f.l.': 0.7968410646160216, 'Average Precision u.f.l.': 0.9212800110948078, 'Precision u.f.l.': 0.9415294873031299}
vgg_abl_la_2_512_3_warm = {'Accuracy u.f.l.': 0.8262874319419238, 'F1 u.f.l.': 0.8891834196115813, 'ROC-AUC u.f.l.': 0.7832083207654539, 'Average Precision u.f.l.': 0.9160185907829794, 'Precision u.f.l.': 0.9327006551407951}
vgg_abl_la_2_512_4_warm = {'Accuracy u.f.l.': 0.8268262250453721, 'F1 u.f.l.': 0.8894495479505367, 'ROC-AUC u.f.l.': 0.7853249436585203, 'Average Precision u.f.l.': 0.9168234899218008, 'Precision u.f.l.': 0.9336904192373168}
vgg_abl_la_2_512_5_warm = {'Accuracy u.f.l.': 0.7970579060798548, 'F1 u.f.l.': 0.8644222977237022, 'ROC-AUC u.f.l.': 0.8104951675856711, 'Average Precision u.f.l.': 0.927330299725275, 'Precision u.f.l.': 0.955443516742688}
vgg_abl_la_2_512_6_warm = {'Accuracy u.f.l.': 0.8031491039019963, 'F1 u.f.l.': 0.8711147157988697, 'ROC-AUC u.f.l.': 0.788053304192415, 'Average Precision u.f.l.': 0.9181769629473955, 'Precision u.f.l.': 0.9406113479643954}
vgg_abl_la_2_512_7_warm = {'Accuracy u.f.l.': 0.8260180353901997, 'F1 u.f.l.': 0.8889260876984838, 'ROC-AUC u.f.l.': 0.7841198952416912, 'Average Precision u.f.l.': 0.9164484558516359, 'Precision u.f.l.': 0.9333525526220585}
vgg_abl_la_2_512_8_warm = {'Accuracy u.f.l.': 0.8262874319419238, 'F1 u.f.l.': 0.8891110837019127, 'ROC-AUC u.f.l.': 0.7843883018172079, 'Average Precision u.f.l.': 0.9164901028783216, 'Precision u.f.l.': 0.9333380678466907}

########### effb3 ############
eff_abl_sup_not_scaled = {'Accuracy s.l.': 0.8222558864265929, 'F1 s.l.': 0.8860498055178415, 'ROC-AUC u.f.l.': 0.7811706640931675, 'Average Precision s.l.': 0.9156947618073822, 'Precision s.l.': 0.9333000252796648}
eff_abl_sup_scale = {'Accuracy s.l.': 0.7981749192059095, 'F1 s.l.': 0.8651596040016539, 'ROC-AUC u.f.l.': 0.8100376463853592, 'Average Precision s.l.': 0.9273146644871455, 'Precision s.l.': 0.9553160401795602}

eff_abl_bal_sup = {'Accuracy s.l.': 0.8036221590909091, 'F1 s.l.': 0.7909899661071914, 'ROC-AUC u.f.l.': 0.8041222841765546, 'Average Precision s.l.': 0.7469921984889095, 'Precision s.l.': 0.8262599419515703}
eff_abl_bal_ir_noise = {'Accuracy u.f.l.': 0.8064630681818182, 'F1 u.f.l.': 0.7946707787176919, 'ROC-AUC u.f.l.': 0.8070277660962586, 'Average Precision u.f.l.': 0.7496757343427043, 'Precision u.f.l.': 0.8274286932302146}
eff_abl_bal_ir_no_noise = {'Accuracy u.f.l.': 0.7625473484848485, 'F1 u.f.l.': 0.7693174998012497, 'ROC-AUC u.f.l.': 0.7676704798550634, 'Average Precision u.f.l.': 0.6901251688350962, 'Precision u.f.l.': 0.7371815846196076}
eff_abl_bal_la_bi_group = {'Accuracy u.f.l.': 0.8068181818181818, 'F1 u.f.l.': 0.7951335655821523, 'ROC-AUC u.f.l.': 0.8073856759029873, 'Average Precision u.f.l.': 0.7500160222666644, 'Precision u.f.l.': 0.8275651717958966}
eff_abl_bal_la_multi_group = {'Accuracy u.f.l.': 0.7969933712121212, 'F1 u.f.l.': 0.7804373832459978, 'ROC-AUC u.f.l.': 0.7961679759217344, 'Average Precision u.f.l.': 0.7392681503393189, 'Precision u.f.l.': 0.826303587191044}

eff_v2_abl_bal_sup = {'Accuracy s.l.': 0.7987689393939394, 'F1 s.l.': 0.7832939773029751, 'ROC-AUC u.f.l.': 0.7980197039269629, 'Average Precision s.l.': 0.7406065873939407, 'Precision s.l.': 0.8253807015346859}
eff_v2_abl_bal_ir_noise = {'Accuracy u.f.l.': 0.8113162878787878, 'F1 u.f.l.': 0.8021624299093136, 'ROC-AUC u.f.l.': 0.813070884434, 'Average Precision u.f.l.': 0.7556735396315294, 'Precision u.f.l.': 0.8264037843795345}
eff_v2_abl_bal_ir_no_noise = {'Accuracy u.f.l.': 0.7997159090909091, 'F1 u.f.l.': 0.7908375562720202, 'ROC-AUC u.f.l.': 0.8003470847089053, 'Average Precision u.f.l.': 0.7392029401550797, 'Precision u.f.l.': 0.8101881961972194}
eff_v2_abl_bal_la_bi_group = {'Accuracy u.f.l.': 0.8068181818181818, 'F1 u.f.l.': 0.7951335655821523, 'ROC-AUC u.f.l.': 0.8073856759029873, 'Average Precision u.f.l.': 0.7500160222666644, 'Precision u.f.l.': 0.8275651717958966}
eff_v2_abl_bal_la_multi_group = {'Accuracy u.f.l.': 0.7937973484848485, 'F1 u.f.l.': 0.7754539512634402, 'ROC-AUC u.f.l.': 0.7930307246848808, 'Average Precision u.f.l.': 0.7371580507148618, 'Precision u.f.l.': 0.8277123204316275}

eff_v2_abl_unbal_ir_noise = {'Accuracy u.f.l.': 0.8264221302177859, 'F1 u.f.l.': 0.8891717675541435, 'ROC-AUC u.f.l.': 0.7847478357001545, 'Average Precision u.f.l.': 0.9166206014005909, 'Precision u.f.l.': 0.9334959915037292}
eff_v2_abl_unbal_ir_no_noise = {'Accuracy u.f.l.': 0.8038225952813067, 'F1 u.f.l.': 0.8715253266211455, 'ROC-AUC u.f.l.': 0.7905069682336859, 'Average Precision u.f.l.': 0.919454365631993, 'Precision u.f.l.': 0.9423179473089229}
eff_v2_abl_unbal_la_bi_group = {'Accuracy u.f.l.': 0.826556828493648, 'F1 u.f.l.': 0.8892630218780198, 'ROC-AUC u.f.l.': 0.7851656487060734, 'Average Precision u.f.l.': 0.91677483568516, 'Precision u.f.l.': 0.9336763470653789}
eff_v2_abl_unbal_la_multi_group = {'Accuracy u.f.l.': 0.7952019056261342, 'F1 u.f.l.': 0.8640884099703452, 'ROC-AUC u.f.l.': 0.7980032980571374, 'Average Precision u.f.l.': 0.9225351534532673, 'Precision u.f.l.': 0.9487891123737245}

def round_dict(dictionary):
    for key, val in dictionary.items():
        dictionary[key] = np.round(val, 2)
    return dictionary

def dict_to_table_row(name, dictionary):
    row = "{} & ".format(name)
    for _, val in dictionary.items():
        row += "{:.2f} & ".format(val)
    row = row[:-2] + "\\\\"
    return row

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

print("vgg_abl_la_2_512_5: ", round_dict(vgg_abl_la_2_512_5_warm.copy()))

print("vgg_abl_la_4_512_5: ", round_dict(vgg_abl_la_4_512_5.copy()))
print("vgg_abl_la_8_512_5: ", round_dict(vgg_abl_la_8_512_5.copy()))
print("vgg_abl_la_16_512_5: ", round_dict(vgg_abl_la_16_512_5.copy()))
print("vgg_abl_la_32_512_5: ", round_dict(vgg_abl_la_32_512_5.copy()))
print("vgg_abl_la_64_512_5: ", round_dict(vgg_abl_la_64_512_5.copy()))
print("vgg_abl_la_126_512_5: ", round_dict(vgg_abl_la_128_512_5.copy()))

print("vgg_abl_la_2_16_5_warm: ", round_dict(vgg_abl_la_2_16_5_warm.copy()))
print("vgg_abl_la_2_32_5_warm: ", round_dict(vgg_abl_la_2_32_5_warm.copy()))
print("vgg_abl_la_2_64_5_warm: ", round_dict(vgg_abl_la_2_64_5_warm.copy()))
print("vgg_abl_la_2_128_5_warm: ", round_dict(vgg_abl_la_2_128_5_warm.copy()))
print("vgg_abl_la_2_256_5_warm: ", round_dict(vgg_abl_la_2_256_5_warm.copy()))
print("vgg_abl_la_2_512_5_warm: ", round_dict(vgg_abl_la_2_512_5_warm.copy()))
print("vgg_abl_la_2_1024_5_warm: ", round_dict(vgg_abl_la_2_1024_5_warm.copy()))

print("vgg_abl_la_2_512_1_warm: ", round_dict(vgg_abl_la_2_512_1_warm.copy()))
print("vgg_abl_la_2_512_2_warm: ", round_dict(vgg_abl_la_2_512_2_warm.copy()))
print("vgg_abl_la_2_512_3_warm: ", round_dict(vgg_abl_la_2_512_3_warm.copy()))
print("vgg_abl_la_2_512_4_warm: ", round_dict(vgg_abl_la_2_512_4_warm.copy()))
print("vgg_abl_la_2_512_5_warm: ", round_dict(vgg_abl_la_2_512_5_warm.copy()))
print("vgg_abl_la_2_512_6_warm: ", round_dict(vgg_abl_la_2_512_6_warm.copy()))
print("vgg_abl_la_2_512_7_warm: ", round_dict(vgg_abl_la_2_512_7_warm.copy()))
print("vgg_abl_la_2_512_8_warm: ", round_dict(vgg_abl_la_2_512_8_warm.copy()))

print("vgg_abl_bal_sup: ", round_dict(vgg_abl_bal_sup.copy()))
print("vgg_abl_bal_ir_noise: ", round_dict(vgg_abl_bal_ir_noise.copy()))
print("vgg_abl_bal_ir_no_noise: ", round_dict(vgg_abl_bal_ir_no_noise.copy()))
print("vgg_abl_bal_la_bi_group: ", round_dict(vgg_abl_bal_la_bi_group.copy()))
print("vgg_abl_bal_la_multi_group: ", round_dict(vgg_abl_bal_la_multi_group.copy()))

print("eff_abl_bal_sup: ", round_dict(eff_abl_bal_sup.copy()))
print("eff_abl_bal_ir_noise: ", round_dict(eff_abl_bal_ir_noise.copy()))
print("eff_abl_bal_ir_no_noise: ", round_dict(eff_abl_bal_ir_no_noise.copy()))
print("eff_abl_bal_la_bi_group: ", round_dict(eff_abl_bal_la_bi_group.copy()))
print("eff_abl_bal_la_multi_group: ", round_dict(eff_abl_bal_la_multi_group.copy()))

print("eff_v2_abl_bal_sup: ", round_dict(eff_v2_abl_bal_sup.copy()))
print("eff_v2_abl_bal_ir_noise: ", round_dict(eff_v2_abl_bal_ir_noise.copy()))
print("eff_v2_v2_abl_bal_ir_no_noise: ", round_dict(eff_v2_abl_bal_ir_no_noise.copy()))
print("eff_v2_abl_bal_la_bi_group: ", round_dict(eff_v2_abl_bal_la_bi_group.copy()))
print("eff_v2_abl_bal_la_multi_group: ", round_dict(eff_v2_abl_bal_la_multi_group.copy()))

print("eff_v2_abl_unbal_ir_noise: ", round_dict(eff_v2_abl_unbal_ir_noise.copy()))
print("eff_v2_abl_unbal_ir_no_noise: ", round_dict(eff_v2_abl_unbal_ir_no_noise.copy()))
print("eff_v2_abl_unbal_la_bi_group: ", round_dict(eff_v2_abl_unbal_la_bi_group.copy()))
print("eff_v2_abl_unbal_la_multi_group: ", round_dict(eff_v2_abl_unbal_la_multi_group.copy()))

print("eff_abl_not_scaled", round_dict(eff_abl_sup_not_scaled.copy()))
print("eff_abl_scale", round_dict(eff_abl_sup_scale.copy()))

print(dict_to_table_row("eff_v2_abl_unbal_ir_noise", round_dict(eff_v2_abl_unbal_ir_noise.copy())))
print(dict_to_table_row("eff_v2_abl_unbal_ir_no_noise", round_dict(eff_v2_abl_unbal_ir_no_noise.copy())))
print(dict_to_table_row("eff_v2_abl_unbal_la_bi_group", round_dict(eff_v2_abl_unbal_la_bi_group.copy())))
print(dict_to_table_row("eff_v2_abl_unbal_la_multi_group", round_dict(eff_v2_abl_unbal_la_multi_group.copy())))