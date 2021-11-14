########https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T


class PennFudanDatest(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #load all image files , sorting them to
        #ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImage"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        #load image and masks
        img_path = os.path.join(self.root, "PNGImage", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        #convert the
        mask = Image.open(mask_path)
        #convert the pil image into a numpy array
        mask = np.array(mask)
        #instance are encode as different colors
        obj_ids = np.unique(mask)
        #first id is background so remove it
        obj_ids = obj_ids[1:]

        #split  the color_encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        #get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, xmax, ymin, ymax])

        #convert everying into a torch.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(labels, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1])*(boxes[:, 2] - boxes[: ,0])
        #suppose all install are not croward

        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

#finetuning from a pretrained model
def test_finetuning():
    #load a model pre-trained on coco
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #repalce the classifier with a new one that has num_class which is user_defined
    num_classes = 2#1 class(person) + background
    #get numbers of input features for the classifier
    in_features = model.roi_heads.box_predictior.cls_score.in_features
    #replace the pretrained head with a new one
    model.roi_heads.box_predition = FastRCNNPredictor(in_features, num_classes)

#add a different backone
def test_add_backbone():
    #load a pre_trained model for classification and return only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    #fastrcnn needs to know the number of output channel in a backbone for mobile_v2 it's 1280
    #so we need to add it here
    backbone.out_channels = 1280
    #let's make the rpn generate 5X3 anchors per spatial location , with 5 different sizes and 3 different aspect rations
    #we have a tuple because each feature map could potentially have different sizes and aspects rations
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) )
    #let's define what are the feature maps that we will

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    #put the pieces together inside a fasterrcnn model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    #instance segmention model for pennfudan dataset
def get_model_instance_segmentation(num_classes):
    #load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    #get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_sore.in_features

    #replace the pre-trianed head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #and replace the mask predicitor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

#putting everything together
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

#teting forward method
def test_forward():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    dataset = PennFudanDatest('PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size =2,
        shuffle = True,
        num_works = 4,
        collate_fn = utils.collate_fn)
    #for training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.item()} for t in targets]
    output = model(images, targets)#return losses and detection
    #for inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)

#main function which perform the training and the validation
from engine import train_one_epoch, evaluate
import utils

def main():
    #train on gpu or cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #our dataset has two classes onlu
    num_classes = 2
    #use our dataset and define transformatation
    dataset = PennFudanDatest('PeenFudanPed', get_transform(train=True))
    dataset_test = PennFudanDatest('PennFudanPed', get_transform(train=False))

    #split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:-50])

    #define traing and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size = 2,
                                              shuffle=True,
                                              num_works=4,
                                              collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_szie=1,
                                                   shuffle = False,
                                                   num_classes=4,
                                                   collate_fn = utils.collate_fn)

    #get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    #move model to the right device
    model.to(device)

    #construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    #let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        #update the learning rate
        lr_scheduler.step()
        #evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


