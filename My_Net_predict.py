

import os
from PIL import Image
from My_Net import My__net

if __name__ == "__main__":
    net = My__net()
    Data_path = 'Data'
    with open(os.path.join(Data_path, "Dataset/ImageSets/Segmentation/test.txt"), "r") as f:
        test_lines = f.readlines()
    num_test = len(test_lines)
    imgsave_path = os.path.join(Data_path, 'Dataset/result_mynet')

    mode = "predict"
    if mode == "predict":
        # while True:
            for i in range(num_test):
                annotation_line = test_lines[i]
                name = annotation_line.split()[0]
                try:
                    image = Image.open(os.path.join(os.path.join(Data_path, "Dataset/JPEGImages"), name + ".jpg"))
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = net.detect_image(image)
                    r_image.save(os.path.join(imgsave_path, name + ".PNG"))
                    # r_image.show()
