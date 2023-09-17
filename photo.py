import argparse
import os
import torch
import cv2
import numpy as np
import argparse
from model.architecture import IRNet_1, IRNet_2
from model.architecture_SRITM import SRITM_IRNet_5


def traverse_under_folder(folder_root):
	folder_leaf = []
	folder_branch = []
	file_leaf = []
	
	index = 0
	for dirpath, subdirnames, filenames in os.walk(folder_root):
		index += 1
	
		if len(subdirnames) == 0:
			folder_leaf.append(dirpath)
		else:
			folder_branch.append(dirpath)
	
		for i in range(len(filenames)):
			file_leaf.append(os.path.join(dirpath, filenames[i]))

	return folder_leaf, folder_branch, file_leaf

testdata_path = './HDRTV_test/test_sdr/'
_,_,fil = traverse_under_folder(testdata_path)
fil.sort()


model_path = "./pretrained_models/SRITM_IRNet-2_True.pth"
model = IRNet_2(upscale=4)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model = model.eval()


for i in range(len(fil)):

    img_path = fil[i]
        
    img = cv2.cvtColor((cv2.imread(img_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().clamp(min=0, max=1)

    data = torch.tensor(img)
    data = torch.unsqueeze(data,0).cpu()

    with torch.no_grad():
        img = model(data)
    #img = model(data)



    img = torch.squeeze(img,0).clamp(min=0, max=1)
    img = img.detach().numpy()


    img = np.transpose(img,(1,2,0))
    img = img.astype(np.float32)
    img = ((img*65535).astype(np.uint16))
    
    
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)



    cv2.imwrite("./results/"+str(i+1).rjust(3,'0')+".png", img)
    print(i)
