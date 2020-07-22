import json
import cv2
import os

from methods.jsonExport.config import localConfig

###############################################################
#class Info
class Info:
    def __init__(self):
        self.year = 2020
        self.version ='0.1'
        self.description = 'LADI AUTOMATIC ANNOTATION'
        self.name = 'CERTH_LADI'

    def printInfo(self):
        print("------------------------------")
        print("year:", self.year)
        print("version:", self.version)
        print("description:", self.description)
        print("name:", self.name)

###############################################################
#class category
class Category:
    def __init__(self):
        self.id = -1
        self.name = ''

###############################################################
#class Categories
class Categories(list):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent='\t')

    def printCategories(self):
        print("------------------------------")
        print("categories:", self.toJSON())

    def fromList(self, _list):
        for index in range(len(_list)):
            tempCategory = Category()
            tempCategory.id = index
            tempCategory.name = _list[index]
            self.append(tempCategory)

###############################################################
#class Image
class Image:
    def __init__(self):
        self.file_name = ''
        self.height = -1
        self.width = -1
        self.path = "Path_to_be_replaced"
        self.id = 1
        self.cls_prob = []

    def fromFileName(self, index,  fileName):
        image = cv2.imread(os.path.join(localConfig.IMAGE_FOLDER, fileName))
        self.file_name = fileName
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.id = index

         # "file_name": "ae47c54ea5ae216488fba19144f6db8f0b19c33e.jpg",
         # "height": 683,
         # "width": 1024,
         # "path": "Path_to_be_replaced",
         # "id": 0
         # "cls_prob": []

    def setClsProb(self, _list):
        self.cls_prob = _list

##################################################################
#class Images
class Images(list):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent='\t')


##############################################################
#class Annotation
class Annotation:
    def __init__(self):
        self.image_id = -1
        self.category_id = -1
        self.id = -1
        self.bbox = []
        self.area = -1
        self.iscrowd = 0
        self.segmentation = []
        self.score = 0

    def setId(self, id):
        self.id = id

        # "image_id": 0,
        # "category_id": 4,
        # "id": 0,
        # "bbox": [
        #     33,
        #     183,
        #     163,
        #     78
        # ],
        # "area": 12714,
        # "iscrowd": 0,
        # "segmentation": []


##############################################################
#class Annotations
class Annotations(list):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent='\t')

    def printAnnotations(self):
        print("------------------------------")
        print("categories:", self.toJSON())

    def fixIds(self):
        for index in range(len(self)):
            self[index].setId(index)


################################################################
#class Dataset
class Dataset:
    def __init__(self):
        self.info = {}
        self.images = []
        self.annotations = []
        self.categories = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)


    def save2JSON(self):
        with open(os.path.join(localConfig.OUTPUT_FOLDER, 'results_output_ladi2.json'), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__, sort_keys=False, indent='\t')
            print("json file saved successfully...")

    def printDataset(self):
        print("------------------------------")
        print("info:", self.info)
        print("images:", self.images)
        print("categories:", self.categories)

