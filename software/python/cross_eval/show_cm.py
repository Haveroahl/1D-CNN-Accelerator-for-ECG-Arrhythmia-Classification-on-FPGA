import json

with open(r'd:\Thesis101\software\python\results\cross_eval\ptbxl_cross_eval.json') as f:
    r = json.load(f)

classes = ['AFIB', 'GSVT', 'SB', 'SR']
modes = ['C2_zeroshot', 'C3_linear_probe', 'C4_full_finetune', 'C5_from_scratch']

for mode in modes:
    m = r[mode]
    acc = m['acc']
    f1  = m['f1_macro']
    f1p = m['f1_per_class']
    cm  = m['confusion_matrix']
    print()
    print('=== ' + mode + ' | acc=' + str(round(acc,4)) + ' f1=' + str(round(f1,4)) + ' ===')
    f1_str = '  '.join(c + ':' + str(round(f1p[i],3)) for i, c in enumerate(classes))
    print('Per-class F1: ' + f1_str)
    print('Confusion matrix (row=true, col=pred):')
    header = '       ' + '  '.join(c.rjust(5) for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = '  ' + classes[i].rjust(4) + ': ' + '  '.join(str(v).rjust(5) for v in row)
        print(row_str)
