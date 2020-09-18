#data augmentation using keras 

from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img

aug = ImageDataGenerator(
      rotation_range = 30,
      width_shift_range= 0.2,
      height_shift_range=0.1,
      zoom_range=0.3,
      horizontal_flip=True,
      vertical_flip=True,
      shear_range=0.2,    
      fill_mode="nearest")
                         

img = load_img('Z:/courses/GITHUB/3_DataAugmentation_DL_Keras/dog.4774.jpg')
x = img_to_array(img)
x.shape
Y= x.reshape((1,) + x.shape)


i = 0
for batch1 , batch2 in aug.flow(Y, batch_size=1 , 
                      save_to_dir = "Z:/courses/GITHUB/3_DataAugmentation_DL_Keras/output", 
                      save_format="jpeg" , save_prefix = "dog"):
   
 i += 1
 if i > 25:
  break
















