from methods.jsonExport import inference
import os
import methods.jsonExport.entities as entities
from methods.jsonExport.config import localConfig

if __name__ == '__main__':
    info =  entities.Info()

    categories = entities.Categories()
    categories.fromList(localConfig.CLASSES)

    img_set = os.listdir(localConfig.IMAGE_FOLDER)

    images = entities.Images()
    annotations = entities.Annotations()

    model = inference.create_model()
    img_id = 0
    for image_file_name in img_set:
        image = entities.Image()
        tempAnnotation = inference.inference_image(img_id, image_file_name, model)
        image.fromFileName(img_id, image_file_name)
        images.append(image)
        annotations.extend(tempAnnotation)
        img_id += 1

    annotations.fixIds()

    dataset = entities.Dataset()
    dataset.info = info
    dataset.categories = categories
    dataset.images = images
    dataset.annotations = annotations
    dataset.save2JSON()
