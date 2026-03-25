from pathlib import Path
import os
from model import inference

labels = ['notumor', 'tumor']
# img_path = Path(r"D:\lokesh dutt\Downloads\Brain Tumor Dataset New\tumor")
img_path = Path(r"D:\lokesh dutt\Downloads\Brain Tumor Dataset New\notumor")

allimg = os.listdir(img_path)
label_count = 0
for inp in range(len(allimg)):
    print(f"<---------------------- For input no. {inp+1} ----------------------->")
    inp_img = allimg[inp]
    pred_class,conf = inference(os.path.join(img_path,inp_img))

    if(pred_class == 0):
        label_count += 1
        print("label count",label_count)
    
    print("pred_class :- ",pred_class)
    print("pred_label :- ",labels[pred_class])
    print("conf :- ",conf*100)

print("Total label :- ",label_count)