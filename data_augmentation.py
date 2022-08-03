from torchvision import transforms

class data_augmentation_transform():

    def transform(self, phase):

         transform_dict = {

            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                #transforms.ColorJitter(brightness=0.15, contrast=0.3, hue=0.2),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # actually belongs to transform not aug.
                          ])                                                             # use mean and std. calculated based on imagenet

            #'test': transforms.Compose([

                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

         #])

    }

         return transform_dict[phase]
 # compress to between 1 and 0 or use image's self mean and std.

