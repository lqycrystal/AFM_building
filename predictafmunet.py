import torch
import numpy as np
import os

import argparse
from unet_model import UNet
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from skimage import io
import os
from sklearn.metrics import confusion_matrix
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"
class ToTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic).float()
        img = img.transpose(0, 1).transpose(0, 2)
        return img  
class generateDataset(Dataset):

        def __init__(self, dirFiles,img_size,colordim,isTrain=True):
                self.isTrain = isTrain
                self.dirFiles = dirFiles
                self.nameFiles = [name for name in os.listdir(dirFiles) if os.path.isfile(os.path.join(dirFiles, name))]
                self.numFiles = len(self.nameFiles)
                self.img_size = img_size
                self.colordim = colordim
                print('number of files : ' + str(self.numFiles))
                
        def __getitem__(self, index):
                filename = self.dirFiles + self.nameFiles[index]
                img = io.imread(filename)
                img = ToTensor()(img)
                imgName, imgSuf = os.path.splitext(self.nameFiles[index])
                return img, imgName
        
        def __len__(self):
                return int(self.numFiles)
def map01(tensor):
    #input/output:tensor
    maxa=np.copy(tensor.numpy())
    mina=np.copy(tensor.numpy())
    maxa[:,0,:,:]=255.0
    maxa[:,1,:,:]=255.0
    maxa[:,2,:,:]=255.0
    maxa[:,3,:,:]=11.65
    maxa[:,4,:,:]=10.98
    mina[:,0,:,:]=0.0
    mina[:,1,:,:]=0.0
    mina[:,2,:,:]=0.0
    mina[:,3,:,:]=-25.38
    mina[:,4,:,:]=-28.50
    return torch.from_numpy( (tensor.numpy() - mina) / (maxa-mina) )
def evaluate(cm):

        UAur=float(cm[1][1])/float(cm[1][0]+cm[1][1])
        UAnonur=float(cm[0][0])/float(cm[0][0]+cm[0][1])
        PAur=float(cm[1][1])/float(cm[0][1]+cm[1][1])
        PAnonur=float(cm[0][0])/float(cm[1][0]+cm[0][0])
        OA=float(cm[1][1]+cm[0][0])/float(cm[1][0]+cm[1][1]+cm[0][0]+cm[1][0])
        F1=2*UAur*PAur/(UAur+PAur)
        IoU=float(cm[1][1])/float(cm[1][0]+cm[1][1]+cm[0][1])
        
        return OA, F1, IoU

def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    model = UNet(n_channels=args.colordim,n_classes=args.num_class)
    model2 = UNet(n_channels=args.colordim2,n_classes=args.num_class2)
    if args.cuda:
      model=model.cuda()
      model2=model2.cuda()
    model.load_state_dict(torch.load(args.pretrain_net))
    model2.load_state_dict(torch.load(args.pretrain_net2))
    model.eval()
    model2.eval()
    predDataset = generateDataset(args.pre_root_dir, args.img_size, args.colordim, isTrain=False)
    predLoader = DataLoader(dataset=predDataset, batch_size=args.predictbatchsize, num_workers=args.threads)
    with torch.no_grad():
      cm_w = np.zeros((2,2))
      for batch_idx, (batch_x, batch_name) in enumerate(predLoader):
        batch_x = batch_x
        if args.cuda:
            batch_x = batch_x.float().cuda()

        out1 = model(batch_x)
        prediction2 = torch.cat((batch_x,out1),1)
        out= model2(prediction2)
        pred_prop, pred_label = torch.max(out, 1)
        pred_label_np = pred_label.cpu().numpy()        
        for id in range(len(batch_name)):
                predLabel_filename = args.preDir + '/' + batch_name[id] + '.png'
                
                pred_label_single = pred_label_np[id, :, :]
                label_filename= args.label_root_dir +  batch_name[id] + '.png'
                label = io.imread(label_filename)
                cm = confusion_matrix(label.ravel(), pred_label_single.ravel())
                pred_label_single=np.where(pred_label_single>0,255,0)
                print(np.max(pred_label_single))
                print(batch_name[id])
                if (np.max(pred_label_single)>0):
                 io.imsave(predLabel_filename, pred_label_single.astype(np.uint8))
                #else:
                 #io.imsave(predLabel_filename, pred_label_single.astype(np.int32))
                 cm_w = cm_w + cm
                #OA_s, F1_s, IoU_s = evaluate(cm)
                #print('OA_s = ' + str(OA_s) + ', F1_s = ' + str(F1_s) + ', IoU = ' + str(IoU_s))
        
      print(cm_w)      
      OA_w, F1_w, IoU_w = evaluate(cm_w)
      print('OA_w = ' + str(OA_w) + ', F1_w = ' + str(F1_w) + ', IoU = ' + str(IoU_w))

# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1,type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=512,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=3,type=int,
                        help="color dimension of the input image")
    parser.add_argument('--colordim2', default=5,type=int,
                        help="color dimension of the input tensor") 
    parser.add_argument('--pretrain_net', default='model1_epoch_115.pth',
                        help='path of saved pretrained model1') 
    parser.add_argument('--pretrain_net2', default='model2_epoch_115.pth',
                        help='path of saved pretrained model2')                        
    parser.add_argument('--pre_root_dir', default='./INRIAafm/urban/val/data/',
                        help='path of input datasets for predict')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_class2', default=2, type=int,
                        help='number of classes of final output')
    parser.add_argument('--preDir', default='./imageafmsegunet',
                        help='path of result')
    parser.add_argument('--label_root_dir', default='./INRIAafm/urban/val/seg/',
                        help='path of label of input datasets')
    args = parser.parse_args()

    if not os.path.isdir(args.preDir):
        os.makedirs(args.preDir)
    main(args)