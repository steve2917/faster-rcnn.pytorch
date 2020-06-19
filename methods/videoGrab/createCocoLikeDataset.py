import json
import os
import numpy as np
from utils import getMask,binary_mask_to_polygon

def findBBOX(poly):
    Xs=[]
    Ys=[]
    for ke in poly.keys():
        if ke.find('x')==0:
            Xs.append(poly[ke])
        if ke.find('y')==0:
            Ys.append(poly[ke])

    X1=min(Xs)
    Y1=min(Ys)
    X2 = max(Xs)
    Y2 = max(Ys)

    return [X1,Y1,X2-X1,Y2-Y1]







paths=['/home/staurid/Desktop/treckVid/outputs_180_Manos/','/home/staurid/Desktop/treckVid/outputs_181_kwstas/','/home/staurid/Desktop/treckVid/outputs1','/home/staurid/Desktop/treckVid/outputs']

with open('DictConvert2.json', 'r') as fp:
    catoldMap=json.load(fp)

with open('catMap2.json', 'r') as fp:
    catMap=json.load(fp)

with open('info.json', 'r') as fp:
    info=json.load(fp)

with open('catFile2.json', 'r') as fp:
    catFile=json.load(fp)


print(len(os.listdir(paths[0]))+len(os.listdir(paths[1])))



imageID=0
imageList1=[]
annotationList1=[]
imageList2=[]
annotationList2=[]
count=0
objectID=0
for path in paths:
    files=os.listdir(path)
    for file in files:
        with open(path+file,'r') as fp:
            data=json.load(fp)
        print(path+file)
        if data['labeled']==False:
            continue

        count+=1
        tImageDict=dict()
        tImageDict['file_name']=file.split('.')[0]+'.jpg'
        tImageDict['height']=data['size']['height']
        tImageDict['width']=data['size']['width']
        tImageDict['path']='Path_to_be_replaced'
        tImageDict['id']=imageID
        if count<=850:
            imageList1.append(tImageDict)
            anns=data['outputs']['object']
            for ann in anns:
                categy=catoldMap[ann['name']]
                cat_id=catMap[categy]
                try:
                    bbox=findBBOX(ann['polygon'])

                except:
                    continue
                annD={'image_id':imageID, 'category_id':cat_id, 'id':objectID, 'bbox':bbox, 'area':bbox[2]*bbox[3],'iscrowd':0}
                annotationList1.append(annD)
                objectID+=1
            imageID+=1
        else:
            imageList2.append(tImageDict)
            anns = data['outputs']['object']
            for ann in anns:
                categy = catoldMap[ann['name']]
                cat_id = catMap[categy]
                try:
                    bbox = findBBOX(ann['polygon'])

                except:
                    continue
                annD = {'image_id': imageID, 'category_id': cat_id, 'id': objectID, 'bbox': bbox, 'area':bbox[2]*bbox[3],'iscrowd':0}
                annotationList2.append(annD)
                objectID += 1
            imageID += 1

finalDataset1=dict()
finalDataset1['info']=info['info']
finalDataset1['images']=imageList1
finalDataset1['annotations']=annotationList1
finalDataset1['categories']=catFile['categories']

with open('finalDatasetTrainNEW.json', 'w') as fp:
    json.dump(finalDataset1,fp, indent=1)

finalDataset2=dict()
finalDataset2['info']=info['info']
finalDataset2['images']=imageList2
finalDataset2['annotations']=annotationList2
finalDataset2['categories']=catFile['categories']

with open('finalDatasetTestNEW.json', 'w') as fp:
    json.dump(finalDataset2,fp, indent=1)

print(count)