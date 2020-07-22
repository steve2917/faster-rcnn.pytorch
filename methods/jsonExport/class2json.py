from methods.jsonExport import inference
import os
import numpy as np
import methods.jsonExport.entities as entities
from methods.jsonExport.config import localConfig

def image_cls_prob(scores_array):

    def max_cords(arr2D):
        #find max value of the array
        maxValue = np.amax(arr2D)
        # Find index of maximum value from 2D numpy array
        result = np.where(arr2D == maxValue)

        return maxValue, int(result[0][0]), int(result[1][0])

    #convert to numpy array
    #removes first dimension (batch dimension)
    #[0,300,22] -> [300,22]
    scores_array = scores_array[0,:,:].cpu().numpy()

    #removes first column (background column)
    scores_array = np.delete(scores_array, 0, axis=1)

    class_num = scores_array.shape[1]
    #initialize output
    output = [0]*class_num

    #loop on classes
    for row in range(class_num):
        maxValue, bbox, classe = max_cords(scores_array)
        output[classe] = float(maxValue)
        scores_array[:, classe] = 0
        scores_array[bbox, :] = 0
    return output

if __name__ == '__main__':
    info = entities.Info()

    categories = entities.Categories()
    categories.fromList(localConfig.CLASSES)

    img_set = os.listdir(localConfig.IMAGE_FOLDER)

    images = entities.Images()
    annotations = entities.Annotations()

    model = inference.create_model()
    img_id = 0

    for image_file_name in img_set:
        image = entities.Image()
        scores = inference.class_scores(img_id, image_file_name, model)
        scores = image_cls_prob(scores)
        #tempAnnotation = inference.inference_image(img_id, image_file_name, model)
        image.fromFileName(img_id, image_file_name)
        image.setClsProb(scores)
        images.append(image)
        #annotations.extend(tempAnnotation)
        img_id += 1

    #annotations.fixIds()

    dataset = entities.Dataset()
    dataset.info = info
    dataset.categories = categories
    dataset.images = images
    #dataset.annotations = annotations
    dataset.save2JSON()



