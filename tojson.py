import json

CLASSES = ['Pipe', 'Car', 'Highway', 'Aircraft', 'Building', 'Railway', 'Rooftop', 'Train', 'Buildings_block',
           'Dam / levee', 'Bridge', 'Power_line', 'Boat', 'Dock', 'Road', 'Container_building',
           'Communication_tower', 'Truck', 'Airway', 'Water_tower', 'Parking_lot']


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


         #
         # "file_name": "ae47c54ea5ae216488fba19144f6db8f0b19c33e.jpg",
         # "height": 683,
         # "width": 1024,
         # "path": "Path_to_be_replaced",
         # "id": 0


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
    categories.printCategories()

    dataset = Dataset()
    dataset.info = info
    dataset.categories = categories
    dataset.save2JSON()
