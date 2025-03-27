from utils.print_utils import plot_lambda_experiments


ir_lambda_experiments = [
    {
        'name': 'IR',
        'lambda': 1e-01,
        'best': {'Accuracy u.f.l.': 0.8125482078039927, 'AUC u.f.l.': 0.7812713160617303, 'F1 u.f.l.': 0.878828694436201, 'Precision u.f.l.': 0.9347465356044318, 'Recall u.f.l.': 0.8298664017368793, 'Average Precision u.f.l.': 0.9154713468743005}
    },
    {
        'name': 'IR',
        'lambda': 1e-02,
        'best': {'Accuracy u.f.l.': 0.8233240698729583, 'AUC u.f.l.': 0.7861000296837495, 'F1 u.f.l.': 0.8867756696117511, 'Precision u.f.l.': 0.934800716640945, 'Recall u.f.l.': 0.8439970020856096, 'Average Precision u.f.l.': 0.917139672774438}
    },
    {
        'name': 'IR',
        'lambda': 1e-03,
        'best': {'Accuracy u.f.l.': 0.8179772572595282, 'AUC u.f.l.': 0.7837735717378294, 'F1 u.f.l.': 0.8828418916404353, 'Precision u.f.l.': 0.934860511621213, 'Recall u.f.l.': 0.8367860716646767, 'Average Precision u.f.l.': 0.9163898259397405}
    },
    {
        'name': 'IR',
        'lambda': 1e-04,
        'best': {'Accuracy u.f.l.': 0.8260180353901997, 'AUC u.f.l.': 0.784107829386639, 'F1 u.f.l.': 0.8889093768488946, 'Precision u.f.l.': 0.9332893703600236, 'Recall u.f.l.': 0.8490314604045118, 'Average Precision u.f.l.': 0.9164187413844654}
    },
    {
        'name': 'IR',
        'lambda': 1e-05,
        'best': {'Accuracy u.f.l.': 0.8242669578039927, 'AUC u.f.l.': 0.7884306574795101, 'F1 u.f.l.': 0.8873514519010135, 'Precision u.f.l.': 0.9359992232981253, 'Recall u.f.l.': 0.8440088294183684, 'Average Precision u.f.l.': 0.9181517761939646}
    },
    {
        'name': 'IR',
        'lambda': 1e-06,
        'best': {'Accuracy u.f.l.': 0.8210341991833031, 'AUC u.f.l.': 0.7812058784658455, 'F1 u.f.l.': 0.8854040221952325, 'Precision u.f.l.': 0.9328788325442953, 'Recall u.f.l.': 0.8430160842135195, 'Average Precision u.f.l.': 0.9154075381465006}
    }
]
la_lambda_experiments = [
    {
        'name': 'LA',
        'lambda': 1e-01,
        'best': {'Accuracy u.f.l.': 0.8245363543557169, 'AUC u.f.l.': 0.7832285441120965, 'F1 u.f.l.': 0.8878534251511245, 'Precision u.f.l.': 0.9331721130129429, 'Recall u.f.l.': 0.8472398321004322, 'Average Precision u.f.l.': 0.916110297128365}
    },
    {
        'name': 'LA',
        'lambda': 1e-02,
        'best': {'Accuracy u.f.l.': 0.8258833371143376, 'AUC u.f.l.': 0.7841445847821028, 'F1 u.f.l.': 0.8888261960344643, 'Precision u.f.l.': 0.9333095497218766, 'Recall u.f.l.': 0.848893496846034, 'Average Precision u.f.l.': 0.9164082862359496}
    },
    {
        'name': 'LA',
        'lambda': 1e-03,
        'best': {'Accuracy u.f.l.': 0.8254792422867514, 'AUC u.f.l.': 0.7841796135599541, 'F1 u.f.l.': 0.8885024048221033, 'Precision u.f.l.': 0.9334324930008219, 'Recall u.f.l.': 0.8482298786864029, 'Average Precision u.f.l.': 0.9164302304346417}
    },
    {
        'name': 'LA',
        'lambda': 1e-04,
        'best': {'Accuracy u.f.l.': 0.7970579060798548, 'AUC u.f.l.': 0.8104951675856711, 'F1 u.f.l.': 0.8644222977237022, 'Precision u.f.l.': 0.955443516742688, 'Recall u.f.l.': 0.790018314891956, 'Average Precision u.f.l.': 0.927330299725275}
    },
    {
        'name': 'LA',
        'lambda': 1e-05,
        'best': {'Accuracy u.f.l.': 0.8192420031760436, 'AUC u.f.l.': 0.7924952505119723, 'F1 u.f.l.': 0.8832280081843839, 'Precision u.f.l.': 0.9390966240166613, 'Recall u.f.l.': 0.8341321880039164, 'Average Precision u.f.l.': 0.9196507542989047}
    },
    {
        'name': 'LA',
        'lambda': 1e-06,
        'best': {'Accuracy u.f.l.': 0.8245363543557169, 'AUC u.f.l.': 0.7850439542432319, 'F1 u.f.l.': 0.8877130970761477, 'Precision u.f.l.': 0.9340997100030213, 'Recall u.f.l.': 0.8462305831688813, 'Average Precision u.f.l.': 0.9167698259954752}
    },

]


plot_lambda_experiments('table', ir_lambda_experiments)
plot_lambda_experiments('table', la_lambda_experiments)