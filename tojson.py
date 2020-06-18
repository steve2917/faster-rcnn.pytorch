import json
import inference
import os
import cv2

CLASSES = ['Pipe', 'Car', 'Highway', 'Aircraft', 'Building', 'Railway', 'Rooftop', 'Train', 'Buildings_block',
           'Dam / levee', 'Bridge', 'Power_line', 'Boat', 'Dock', 'Road', 'Container_building',
           'Communication_tower', 'Truck', 'Airway', 'Water_tower', 'Parking_lot']
#IMAGE_FOLDER = 'images/'
IMAGE_FOLDER = 'data/ladi/images/train/'

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

    def fromFileName(self, index,  fileName):
        image = cv2.imread(os.path.join(IMAGE_FOLDER, fileName))
        self.file_name = fileName
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.id = index


         #
         # "file_name": "ae47c54ea5ae216488fba19144f6db8f0b19c33e.jpg",
         # "height": 683,
         # "width": 1024,
         # "path": "Path_to_be_replaced",
         # "id": 0
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
        with open('results_output_ladi.json', 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__, sort_keys=False, indent='\t')
            print("json file saved successfully...")

    def printDataset(self):
        print("------------------------------")
        print("info:", self.info)
        print("images:", self.images)
        print("categories:", self.categories)


if __name__ == '__main__':

    info = Info()

    categories = Categories()
    categories.fromList(CLASSES)

    img_set = os.listdir(IMAGE_FOLDER)

    images = Images()
    annotations = Annotations()

    model = inference.create_model()
    img_id = 0
    for image_file_name in img_set:
        image = Image()
        tempAnnotation = inference.inference_image(img_id, image_file_name, model)
        image.fromFileName(img_id, image_file_name)
        images.append(image)
        annotations.extend(tempAnnotation)
        img_id += 1


    #annotations = inference.inference_image('0d03b6d82a35626743587f549cf11f4619336d10.jpg')
    annotations.fixIds()

    dataset = Dataset()
    dataset.info = info
    dataset.categories = categories
    dataset.images = images
    dataset.annotations = annotations
    dataset.save2JSON()
