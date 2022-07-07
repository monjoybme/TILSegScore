import os
from keras_segmentation.models.unet import unet, vgg_unet, resnet50_unet
from keras_segmentation.models.fcn import fcn_8, fcn_8_vgg, fcn_8_resnet50
from keras_segmentation.predict import model_from_checkpoint_path, predict_multiple, evaluate


# Params to change
epoch_num = 80
model_name = "fcn8_vgg_aug"
sav_model = "TIL_files/" + model_name + ".h5"
dat_dir = "wsi_data/tissue-cells/split-data/"
Aug = True

# Construct model
if model_name == "unet_vanilla":
 model = unet(n_classes=8,  input_height=512, input_width=512)
elif model_name == "unet_vgg_aug":
 model = vgg_unet(n_classes=8,  input_height=512, input_width=512)
elif model_name == "unet_resnet":
 model = resnet50_unet(n_classes=8,  input_height=512, input_width=512)
elif model_name == "fcn8_vanilla":
 model = fcn_8(n_classes=8,  input_height=512, input_width=512)
elif model_name == "fcn8_vgg_aug":
 model = fcn_8_vgg(n_classes=8,  input_height=512, input_width=512)


# Checkpoint
checkpointPath = os.path.join("TIL_files/weights_{0}/{0}".format(model_name))

# Load model from checkpoint if necessary
#model = model_from_checkpoint_path("tmp/weights/" + model_name")

# Train
print('Using ' + model_name)
print('Saving to ' + checkpointPath)

model.train(
    checkpoints_path = checkpointPath,
    train_images =  dat_dir + "img_train/",
    train_annotations = dat_dir + "annot_train/",
    validate = True,
    val_images =  dat_dir + "img_val/" ,
    val_annotations = dat_dir + "annot_val/",
    verify_dataset=False,
    #load_weights="weights/vgg_unet_1.0" ,
    optimizer_name='adam', 
    do_augment=Aug, 
    augmentation_name="aug_all",    
    epochs = epoch_num,
    steps_per_epoch=512,
    val_steps_per_epoch=512
)

# Model info & save:
#print("Model output shape:", model.output_shape)
#model.summary()
model.save(sav_model)

# Predict
predict_multiple( 
  checkpoints_path=checkpointPath, 
  inp_dir = dat_dir + "img_test/", 
  out_dir="TIL_files/predictions/" + model_name + "_pred/" 
)


# Evaluate model
performance = evaluate(checkpoints_path=checkpointPath, 
                       inp_images_dir=dat_dir + "img_test/", 
		       annotations_dir=dat_dir + "annot_test/" 
)

print(performance)








