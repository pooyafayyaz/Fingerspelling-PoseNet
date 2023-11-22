from model import *
from data_loader import HandPoseDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import SubsetRandomSampler
import json
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from collections import Counter
from torch import nn
import sys
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import numpy as np
from torchinfo import summary
from text_ctc_utils import * 
from utils import *
from ctc_decoder import Decoder
from lm_scorer import Scorer
from torchvision.ops.focal_loss import sigmoid_focal_loss
import kornia.losses
from torchvision import transforms
from gaussian_noise import GaussianNoise

# data_dir = "/home/negar/Desktop/Pooya/TF-DeepHand/mediapipe_res_chicago/"
# hand_detected_label = "/home/negar/Desktop/Pooya/Self_Supervised_ASL_Finger_Spelling/sign_hand_detection.csv"
# labels_csv = "/home/negar/Documents/Datasets/ChicagoFSWild/ChicagoFSWild.csv"

data_dir = "/home/negar/Documents/Datasets/ChicagoWild++/mediapipe_res_chicago/"
hand_detected_label = "/home/negar/Desktop/Pooya/TF-DeepHand/Transformer/sign_hand_detection_wild++_first.csv"
labels_csv = "/home/negar/Desktop/Pooya/TF-DeepHand/Transformer/final.csv"

batch_size = 1
num_workers = 10
char_counts = 32
learning_rate = 0.0001
optim_step_size = 10
optim_gamma = 0.1
num_epochs = 120
SOS_token = 32
EOS_token = 0

#decode_type = "beam"
decode_type = "trans"

beam_size  = 5
lm_beta = 0.4
ins_gamma = 1.2
chars = "$' &.@acbedgfihkjmlonqpsrutwvyxz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print ('device', device)

vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)
decoder_dec = Decoder(char_list, blank_index=0)

print(vocab_map)
print(inv_vocab_map)
print(char_list)

target_enc_df = convert_text_for_ctc(labels_csv,vocab_map,True)

transform = transforms.Compose([GaussianNoise()])

dataset_train = HandPoseDataset(data_dir, labels_csv ,hand_detected_label, target_enc_df , "train", transform=transform)
traindataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=num_workers)

dataset_test = HandPoseDataset(data_dir, labels_csv , hand_detected_label, target_enc_df , "test" , augmentations =False )
testdataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers)

model = TransformerModel(output_dim=len(char_list), d_input = 42 ,d_model=256, nhead=8, num_layers=3, dropout=0.1).to(device)
# model.load_state_dict(torch.load('/home/negar/Desktop/Pooya/TF-DeepHand/Transformer/best_model_62.7970.pt'))


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

loss_encoder = nn.CTCLoss(blank=0,zero_infinity=True, reduction='none')
loss_decoder = nn.CrossEntropyLoss()
# loss_decoder = nn.NLLLoss()
loss_cls = torch.nn.MSELoss()


# loss_decoder = sigmoid_focal_loss()

scheduler = StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

vocab_map_enc, inv_vocab_map_enc, char_list_enc = get_ctc_vocab(chars[1:])
decoder_enc = Decoder(char_list_enc, blank_index=0)
print(vocab_map_enc)
print(inv_vocab_map_enc)
print(char_list_enc)


best_acc = 0

for epoch in range(num_epochs):
    total_loss = 0
    total_loss_cls = 0

    model.train()

    for i, (poses, labels) in enumerate(traindataloader):
        optimizer.zero_grad()
                
        cls_token , logits_lm, encoder_out = model(poses.to(device), labels[:, :-1].to(device))

        log_probs_enc = F.log_softmax(encoder_out, dim=-1).permute(1,0,2)
        
        input_lengths = torch.full((encoder_out.size(0),), log_probs_enc.size(0), dtype=torch.long)
        target_lengths = torch.full((encoder_out.size(0),), labels.size(1)-2, dtype=torch.long )
        
        loss_enc = loss_encoder(log_probs_enc, labels[:,1:-1], input_lengths=input_lengths, target_lengths=target_lengths)
        
        # print(log_probs_enc.device,input_lengths.device)

        # loss_enc = ctcloss_reference(log_probs_enc, labels[:,1:-1].cuda(), input_lengths, target_lengths, logits_lm= logits_lm[0,:-1]).float()
        
        # print(loss_enc,expected)

        loss_dec = loss_decoder(logits_lm[0].cpu(), labels[:,1:].view(-1))


        # if i%30 == 0:
        #     current_preds_enc = decoder_enc.greedy_decode(log_probs_enc[:,0,:].detach().cpu().numpy())
        #     current_preds_enc = ''.join(current_preds_enc)
        #     print(current_preds_enc, ''.join(invert_to_chars(labels[:,1:-1],inv_vocab_map)))
        
        gt_label_size = torch.tensor([[math.sin(((labels.size(1)-2)/30-0.5)*2*torch.pi),math.cos(((labels.size(1)-2)/30-0.5)*2*torch.pi)]],device = device )
        loss_token = (cls_token, gt_label_size) 

        loss = loss_dec+ 5*loss_enc + loss_token

        if i%400 == 0:    
            print('Epoch {}/{} - loss: {:.4f} - loss_cls: {:.4f}' .format(epoch+1, num_epochs, total_loss/(i+1), total_loss_cls/(i+1)))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_cls += loss_token.item()

    scheduler.step()
    model.eval()
    preds = []
    gt_labels = []

    if epoch < 9 :
        continue
    

    for i, (poses, labels) in enumerate(testdataloader):
        poses = poses.to(device)
        cls_token , logits = model(poses)
        log_probs = F.softmax(logits, dim=-1)

        pred_size = (torch.atan2(torch.tensor([cls_token[0,0].detach().cpu()]),torch.tensor([cls_token[0,1].detach().cpu()]))/(2 * torch.pi) +0.5) * 30
        pred_size = torch.round(pred_size)

        current_preds = decoder_dec.beam_decode_trans(log_probs[0].detach().cpu(), beam_size, model, poses , beta=lm_beta, gamma=ins_gamma)
        current_preds = ''.join(current_preds)

        preds.append(current_preds)

        print(current_preds,''.join(invert_to_chars(labels[:,1:-1],inv_vocab_map)), "   ", pred_size) 
        gt_labels.append(''.join(invert_to_chars(labels[:,1:-1],inv_vocab_map)))

    lev_acc = compute_acc(preds, gt_labels)
    if best_acc < lev_acc:
        best_acc = lev_acc
        torch.save(model.state_dict(), 'best_model.pt')


    print('Epoch {}/{} - Letter Acc: {:.4f} - Best Acc {:.4f}'.format(epoch+1, num_epochs, lev_acc, best_acc))
