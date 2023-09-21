#!/usr/bin/python2.7
from ctypes import alignment
import os
import random
from re import S
import time
from collections import defaultdict
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import json
from eval import update_metrics
from losses import AttentionLoss, FrameWiseLoss, SegmentLossAction
from transformers_models import uvast_model
from utils import Metrics, get_grad_norm, params_count, softmax, write_metrics, refine_transcript, write_metrics_mof_f1, write_return_stats
import pickle
from evaluation.accuracy import Accuracy
from evaluation.f1_score import F1Score
import matplotlib.pyplot as plt
from util_functions import compute_euclidean_dist, plot_confusion_matrix, plot_segm
from  tot import generate_optimal_transport , get_complete_cost_matrix , get_cost_matrix, viterbi_inner
grad_history=[]      
torch.manual_seed(42)  
class Trainer:
    def __init__(self, args):
        self.model = uvast_model(args)
        if torch.cuda.is_available():
            self.model.cuda()
        self.args = args
        
        print('params count:', params_count(self.model))

        # initialize losses
        self.frame_wise_loss = FrameWiseLoss(args)
        self.segment_wise_loss = SegmentLossAction(args)
        self.attn_action_loss = AttentionLoss(args)
        self.model_dir = args.model_dir
        self.q_dict= {}

    def train(self, args, device=None, trainloader=None, testloader=None, testing_dataloader=None):
        
        torch.set_default_dtype(torch.float64)
        
       
        # for training encoder only 
        if not args.use_transcript_dec:
            for name, p in self.model.dec_action.named_parameters():
                p.requires_grad = False
            for name, p in self.model.prediction_action.named_parameters():
                p.requires_grad = False
            for name, p in self.model.pos_embed.named_parameters():
                p.requires_grad = False
            for name, p in self.model.dec_embedding.named_parameters():
                p.requires_grad = False
       
        if args.use_transcript_dec : 
            #when using transcript decoder load model from stage 1
            print("model path : " , args.pretrained_model)
            self.model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'), strict=False)
            print (self.model)
        

        self.model.train()
        self.model.to(device)



        eval_data= {}
        if not self.args.inference_only:
            if args.optimizer == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=self.args.weight_decay)    
            
            elif args.optimizer == 'adamw':
                optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=self.args.weight_decay)

            lr_scheduler = None
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma, verbose=True)
            if self.args.lr_scheduler:
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, mode="min", verbose=True)
                print("use ReduceLSOnPlateau scheduler")
        
        
        for epoch in range(args.num_epochs):
       
            print('epoch', epoch, flush=True)
            if args.inference_only:
                self.inference(testing_dataloader, testloader, epoch + 1, device)
                break
            start = time.process_time() 
            epoch_loss = self.train_one_epoch(trainloader, optimizer, epoch, device)
            end = time.process_time()
            if args.do_timing:
                print('time:', end - start)

            
            if (epoch+1)% args.inference_epochs ==0 :
                if epoch+1 == args.num_epochs:
                    self.inference(testing_dataloader, testloader, epoch + 1, device , is_final= True , eval_data= eval_data) 
                else:
            
                    eval_data = self.inference(testing_dataloader, testloader, epoch + 1, device , eval_data= eval_data)
            
            
            if self.args.lr_scheduler:
                lr_scheduler.step(epoch_loss)
            else:
                lr_scheduler.step()  


    def train_one_epoch(self, trainloader, optimizer, epoch, device):
        
        self.model.train()
        self.frame_wise_loss.reset()
        self.segment_wise_loss.reset()
        self.attn_action_loss.reset()

        epoch_loss = 0
        clip_gradient_value = 0.0
        optimizer.zero_grad()
        
        
        for index, data in enumerate(trainloader):
            framewise_losses = torch.tensor(0.0)
            framewise_losses_g = torch.tensor(0.0)
            segwise_losses = torch.tensor(0.0)
            attn_action_losses = torch.tensor(0.0)
            attn_duration_losses = torch.tensor(0.0)

            self.model.zero_grad(set_to_none=True)
            optimizer.zero_grad() 

            feat,  mask = data['feat'],  data['mask']
            name = data['name']
            
            feat= feat.to(torch.float64).to(device)
            batch_input, mask = feat.to(device),  mask.to(device)
            seg_data = None
            
            predictions_framewise, pred_transcript, pred_crossattn, frames_to_segment_assignment = self.model(batch_input, 
                                                                            mask, seg_data, 
                                                                            no_split_data=None , 
                                                                            mode='train')
            proto_scores= predictions_framewise[0][0]
            #C, N
            with torch.no_grad():
                p_gauss = get_complete_cost_matrix(vid_len= feat.shape[2] , num_clusters=self.args.num_classes , num_videos=self.args.num_videos , sigma=self.args.sigma)
                    #N,C
                q = generate_optimal_transport(proto_scores, self.args.epsilon, p_gauss.transpose() )
                    #C,N
                    
            
            proto_probs = F.softmax(proto_scores / self.args.gaussian_temperature)
            proto_probs = torch.clamp(proto_probs, min=1e-30, max=1)
            
            pseudolabels = torch.tensor(np.argmax(p_gauss,1)).to(device)
            # apply losses
            if self.args.do_framewise_loss_gauss:
                framewise_losses = self.frame_wise_loss(predictions=proto_probs,q=q,  batch_target=pseudolabels, mask=mask, epoch=epoch)
                
                if self.args.log_wandb:
                    wandb.log({"Framewise loss": framewise_losses})
            if self.args.use_transcript_dec:
                
                segments_dict = self.convert_labels_to_segments(pseudolabels)
                
                seg_gt_no_split=torch.unsqueeze( segments_dict['seg_gt'][0][1:-1].to(device) , 0)
                seg_dur_no_split= torch.unsqueeze( segments_dict['seg_dur'][0][1:-1].to(device),0)
                
                
                clusters_idx= np.argmax(q.clone().detach().cpu().numpy(), axis= 1)
                
                sorted_idx= sorted(clusters_idx)
                segment_order= []
                transcript =[]
                for id in sorted_idx: 
                    segment_order.append(np.where(clusters_idx==id)[0][0]+2) #added 2 for sos eos order
                    transcript.append(np.where(clusters_idx==id)[0][0])

                #for fixed order T    
                
                if self.args.fixorder_T: 
                    clusters_idx = [x+2 for x in range(p_gauss.shape[1])]
                    segment_order =clusters_idx
                
                segment_order.insert(0, 0.) #sos
                segment_order.append(-1.)   #eos
                segment_len= len(segment_order)
                segment_order= torch.Tensor([segment_order]).to(device).to(torch.float32)
                #seg_data = (segment_order , non_list)
                seg_gt_act= segment_order
                q_segment= None
                seg_gt_act_loss = F.pad(seg_gt_act.clone()[:, 1:], pad=(0, 1), mode='constant', value=-1)
                
                
                p_gauss_new = get_complete_cost_matrix(vid_len= feat.shape[2] , num_clusters=self.args.num_classes , num_videos=self.args.num_videos , sigma=0.75)
                
                new_p= np.zeros_like(p_gauss_new)
                
                for i, j in enumerate(transcript):
                    new_p[:,j] = p_gauss_new[:,i]
    
                pseudolabels = torch.tensor(np.argmax(new_p,1)).to(device)
                
                # generarting gt for attntion maps
                #(1,C,N)
                # attn_mask_gt = torch.zeros(seg_dur_no_split.shape[0], seg_dur_no_split.shape[1], int(seg_dur_no_split.sum().item())) 
                # seg_cumsum = torch.cumsum(seg_dur_no_split, dim=1)
                # for i in range(seg_dur_no_split.shape[1]):
                #     if i > 0:
                #         attn_mask_gt[0, i, int(seg_cumsum[0, i - 1].item()):int(seg_cumsum[0, i].item())] = 1
                #     else:
                #         attn_mask_gt[0, i, :int(seg_cumsum[0, i].item())] = 1
                # attn_mask_gt_dur = attn_mask_gt.to(batch_input.device)
                
                if self.args.do_segwise_loss or self.args.do_segwise_loss_g or self.args.do_segwise_loss_gauss:
                    segwise_losses = self.segment_wise_loss(pred_transcript, seg_gt_act_loss,batch_input.shape[-1],q_segment,epoch)
                    if self.args.log_wandb:
                        wandb.log({"Segmentwise loss": segwise_losses})
                if self.args.do_crossattention_action_loss_nll: 
                    with torch.no_grad():
                        q_a = generate_optimal_transport(proto_scores, self.args.epsilon, new_p.transpose() )
                        pseudolabels = torch.tensor(np.argmax(q_a.cpu().numpy(),0)).to(device)

                    pseudolabels = torch.unsqueeze(pseudolabels, 0).to(device) # using this to filter background in CA loss. 
                   
                    attn_action_losses = self.attn_action_loss(pred_crossattn, q_a.unsqueeze(0) , pseudolabels , q)
                    if self.args.log_wandb:
                        wandb.log({"Cross attention loss": attn_action_losses})        
            
            
            loss = framewise_losses + segwise_losses + attn_action_losses  
            loss.backward()

            if self.args.adap_clip_gradient:
                obs_grad_norm = get_grad_norm(self.model)
                grad_history.append(obs_grad_norm)
                clip_gradient_value = max(np.percentile(grad_history, self.args.clip_percentile), 0.1)
                if clip_gradient_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_gradient_value)
                    
            
            if epoch < self.args.freeze_epochs:

                for name, p in self.model.named_parameters():
                    if "enc_feat.enc.conv_out" in name or "prototype" in name:
                        
                        p.grad= None
                    
            
            optimizer.step() 
            epoch_loss += loss.item()
        
            
        
        if (epoch+1) % self.args.checkpoint_epoch ==0 :    
            torch.save(self.model.state_dict(), self.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), self.model_dir + "/epoch-" + str(epoch + 1) + ".opt")
        
        # log training losses 
        if self.args.do_framewise_loss or self.args.do_framewise_loss_g or self.args.do_framewise_loss_gauss:
            self.frame_wise_loss.log_metrics(mode="train_framewise", epoch=epoch + 1)
        if self.args.do_segwise_loss or self.args.do_segwise_loss_g or self.args.do_segwise_loss_gauss:
            self.segment_wise_loss.log_metrics(mode="segment_framewise", epoch=epoch + 1)
        if self.args.do_crossattention_action_loss_nll: 
            self.attn_action_loss.log_metrics(mode="crossattention action", epoch=epoch + 1)
        
        if (epoch+1)%self.args.save_plot_epochs==0 and self.args.save_plot :
            
            Q_path =os.path.join(self.args.parent_dir, self.args.experiment_path, self.args.exp_name ,"plots","q")
            if not os.path.exists(Q_path):
                os.mkdir(Q_path)
            Proto_path =os.path.join(self.args.parent_dir, self.args.experiment_path, self.args.exp_name ,"plots","proto_scores")
            if not os.path.exists(Proto_path):
                os.mkdir(Proto_path)
            Proto_probs_path =os.path.join(self.args.parent_dir, self.args.experiment_path, self.args.exp_name ,"plots","proto_probs")
            if not os.path.exists(Proto_probs_path):
                os.mkdir(Proto_probs_path) 
            
            seg_path =os.path.join(self.args.parent_dir, self.args.experiment_path, self.args.exp_name ,"plots","segment")
            if not os.path.exists(seg_path):
                os.mkdir(seg_path)
            cross_path =os.path.join(self.args.parent_dir, self.args.experiment_path, self.args.exp_name ,"plots","frame_to_segment")
            if not os.path.exists(cross_path):
                os.mkdir(cross_path)
            
            
            img_finalq = plot_confusion_matrix(q.clone().detach().cpu().numpy().transpose(),os.path.join(Q_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_Q" ) )
            img_protos = plot_confusion_matrix(proto_scores.clone().detach().cpu().numpy().transpose(),os.path.join(Proto_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_proto_scores" ) )
            img_proto_probs = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy().transpose(), os.path.join(Proto_probs_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_proto_probs" ) )
            
            if self.args.use_transcript_dec:
                img_seg = plot_confusion_matrix(pred_transcript[0][0].clone().detach().cpu().numpy().transpose(), os.path.join(seg_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_segment" ) )
                fs_seg = plot_confusion_matrix(pred_crossattn[0][0].clone().detach().cpu().numpy().transpose(), os.path.join(cross_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_cross" ) )
            
            if self.args.log_wandb:
                wandb.log({"Q Matrix Epoch-{}".format(epoch): wandb.Image(img_finalq)})
                wandb.log({"Dot Product Matrix Epoch-{}".format(epoch): wandb.Image(img_protos)})
                wandb.log({"P Matrix Epoch-{}".format(epoch): wandb.Image(img_proto_probs)})
            #img_dists = plot_confusion_matrix(dists.detach().cpu().numpy(),os.path.join(img_dist_path,"e_"+str(epoch+1)+"_v_"+str(index+1)+"_img_dist") )
           
        if self.args.adap_clip_gradient:
            print('clip_gradient {:.3f}'.format(clip_gradient_value))
        return epoch_loss        
        

    def inference(self, testing_dataloader, testloader, epoch, device , is_final = False, eval_data= None):
        
        actions_dict_inv = {v: k for k, v in testing_dataloader.actions_dict_call.items()}
        self.model.eval()
        num_classes=self.args.num_classes
     

        metrics_seg_viterbi = Metrics()
        accuracy  = Accuracy(self.args)
        gt_list = []
        prediction_list = []
        segmentation_map_list = []
        video_names = []
        frame_count_list=list()
        embedding_list = []
        viterbi_label_list = []
        return_stat = {}
        total_fg_mask=[]
        long_rt= []
        cluster_labels_all =[]
        # if self.args.bg: 
        #     total_fg_mask = np.zeros(len())

        with torch.no_grad():
            self.model.to(device)
            if self.args.inference_only:
                if self.args.path_inference_model:
                    path_to_model = self.args.path_inference_model
                    self.args.results_dir = os.path.dirname(path_to_model)
                else:
                    pretrained_name_for_testing = self.args.exp_name
                    path_to_model = self.args.experiment_path + pretrained_name_for_testing + \
                                '/model/' + self.args.dataset + '/epoch-' + str(self.args.epoch_num_for_testing) + '.model'
                print('LOADING the model {}'.format(path_to_model))

                self.model.load_state_dict(torch.load(path_to_model, map_location="cpu"), strict=True)
            

            for index, data in enumerate(tqdm(testloader, desc="testing epoch {}".format(epoch), leave=False)):
                feat, gt, gt_org, mask , temporal_gt , segmentation_map ,video_name = data['feat'], data['gt'], data['gt_org'], data['mask'] , data['temporal_gt'] , data['segmentation_maps'], data['name']
                batch_input, batch_target, batch_target_org, mask  = feat.to(device), gt.to(device), gt_org.to(device), mask.to(device)
                batch_input= batch_input.to(torch.float64).to(device)
                batch_target= batch_target.to(torch.float64).to(device)
                temporal_gt= data['temporal_gt']
                temporal_gt=temporal_gt.to(device)
                
                num_frames= feat.shape[2]
                fg_mask =np.ones(num_frames, dtype=bool)
                if self.args.bg :
                    indexes = [i for i in range(num_frames) if i % 2]
                    fg_mask[indexes] = False
                
                gt_cls_names = []
                for i in range(gt_org.shape[1]):
                    gt_cls_names.extend([actions_dict_inv[batch_target_org[:, i].item()]])

                pred_framewise, pred_transcript, pred_dur, pred_dur_AD, pred_transcript_AD = self.model(batch_input, mask , mode= 'test')
               
                assert batch_input.shape[0] == 1 # we only evaluate one sample at a time
                
                pred_probs = torch.transpose(pred_framewise[0][0],0,1)

                if self.args.use_transcript_dec:
                    with torch.no_grad():
                       
                        p_gauss = get_complete_cost_matrix(vid_len= feat.shape[2] , num_clusters=self.args.num_classes , num_videos=self.args.num_videos , sigma=self.args.sigma)
                        q = generate_optimal_transport(torch.transpose(pred_probs,0,1), self.args.epsilon, p_gauss.transpose() )
            
                    clusters_idx= np.argmax(q.clone().detach().cpu().numpy(), axis= 1)

                    sorted_idx= sorted(clusters_idx)

                    transcript =[]
                    for id in sorted_idx: 

                        transcript.append(np.where(clusters_idx==id)[0][0])

                _, predicted_framewise = torch.max(pred_probs, 1)
                predicted_framewise= torch.unsqueeze(predicted_framewise, 0)
                cluster_labels=predicted_framewise[0][fg_mask]
                cluster_labels=cluster_labels.cpu()
              
                
                if len(long_rt) ==0: 
                    long_rt= temporal_gt.cpu().numpy()
                else: 
                    long_rt= np.append(long_rt,temporal_gt.cpu().numpy())

                if len(total_fg_mask) ==0: 
                    total_fg_mask= fg_mask
                else: 
                    total_fg_mask= np.append(total_fg_mask,fg_mask)

                if len(cluster_labels_all) ==0: 
                    cluster_labels_all= cluster_labels
                else: 
                    cluster_labels_all= np.append(cluster_labels_all,cluster_labels)

                frame_count_list.append(len(predicted_framewise[0]))
                
                video_names.append(video_name)
                if len(prediction_list) ==0: 
                    prediction_list= predicted_framewise.cpu().numpy()
                else: 
                    prediction_list= np.append(prediction_list,predicted_framewise.cpu().numpy())
                
                if len(gt_list) ==0:
                    gt_list = batch_target_org.cpu().numpy()
                else:
                    gt_list= np.append(gt_list, batch_target_org.cpu().numpy())

                segmentation_map_list.append(segmentation_map)

                # transcript predictions
                if self.args.use_transcript_dec:
                    pred_seg_expanded = self.convert_segments_to_labels(pred_transcript, pred_dur, feat.shape[-1])
                                  
                    pred_seg_expanded = torch.clamp(pred_seg_expanded, min=0, max=self.args.num_classes)
                    
                    recog_seg = self.convert_id_to_actions(pred_seg_expanded, gt_org, actions_dict_inv)
                    #update_metrics(recog_seg, gt_cls_names, metrics_segmentwise)
            
                scores = pred_probs.cpu().numpy()
                probs = softmax(scores / 0.1)
                probs = np.clip(probs, 1e-30, 1) 
                log_probs = np.log(probs)
                pi = list(range(self.args.num_classes ))
                if self.args.use_transcript_dec:
                     pi= transcript
                
                # APPLYING VITERBI
                if np.max(log_probs)>0:
                    log_probs= log_probs - 2*(np.max(log_probs))
                
                alignment, return_score=viterbi_inner(log_probs ,probs.shape[0] ,pi )
                #print( " Transcript : ", pi )
                if len(viterbi_label_list)==0:
                    viterbi_label_list = alignment
                else:
                    viterbi_label_list = np.append(viterbi_label_list, alignment)
                    #viterbi_label_list = np.append(viterbi_label_list, pred_seg_expanded[0])
                #update_metrics(alignment, gt_cls_names, metrics_seg_viterbi)
                
                
                
             
           
            print("Total unique clusters: {}".format(np.unique(prediction_list)))
            count_array = np.array(np.unique(prediction_list, return_counts=True)).T
            print("Counts of individual clusters: {}".format(count_array))
            
            time2label = {}
            for label in np.unique(cluster_labels_all):
        
                cluster_mask = cluster_labels_all == label
                r_time = np.mean(long_rt[total_fg_mask][cluster_mask])
                time2label[r_time] = label
        
            for time_idx, sorted_time in enumerate(sorted(time2label)):
                label = time2label[sorted_time]
                cluster_labels_all[cluster_labels_all == label] = time_idx

            shuffle_labels = np.arange(len(time2label))
            print('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
            labels_with_bg = np.ones(len(total_fg_mask)) * -1
            labels_with_bg[total_fg_mask] = cluster_labels_all
            accuracy.predicted_labels = labels_with_bg
            accuracy.gt_labels = gt_list
            if self.args.bg:
                # enforce bg class to be bg class
                accuracy.exclude[-1] = [-1]
           
            old_mof, total_fr = accuracy.mof()

            label2gt ={}
            gt2label = accuracy._gt2cluster
            for key, val in gt2label.items():
                try:
                    label2gt[val[0]] = key
                except IndexError:
                    pass
            print("label to gt: " , label2gt)
            print('MoF val: ' + str(accuracy.mof_val())) # mof without bg
            print('old MoF val: ' + str(float(old_mof) / total_fr))


            # Equivalent to Accuracy corpus after generating prototype likelihood
            f1_score = F1Score(K=num_classes, n_videos=len(testloader))
            gt2label = accuracy._gt2cluster
            accuracy  = Accuracy(self.args)
            accuracy.gt_labels = gt_list
            accuracy.predicted_labels = prediction_list
            f1_score.set_gt(gt_list)
            f1_score.set_pr(prediction_list)
            f1_score.set_gt2pr(gt2label)

            if self.args.bg:
              
                accuracy.exclude[-1] = [-1]
                f1_score.set_exclude(-1)
            print(" second") 
            print(len(accuracy.exclude))  
            old_mof, total_fr = accuracy.mof(old_gt2label=gt2label)
            gt2label = accuracy._gt2cluster

           
            label2gt = {}
            for key, val in gt2label.items():
                try:
                    label2gt[val[0]] = key
                except IndexError:
                    pass
            acc_cur = accuracy.mof_val()
            print(f"MoF val: {acc_cur}")
            print(f"previous dic -> MoF val: {float(old_mof) / total_fr}")
            average_class_mof = accuracy.mof_classes()
            return_stat = accuracy.stat()       
            print("MOF STATS: " , return_stat)
            
            f1_score.f1()
            for key, val in f1_score.stat().items():
                print("key :  " , key , "val : " , val)
                return_stat[key] = val
            start_idx = 0
            for i in range(len(frame_count_list)):
                segmentation_map_list[i]['cl'] = (prediction_list[start_idx : start_idx+frame_count_list[i]] , label2gt)
                start_idx = start_idx + frame_count_list[i]

            
            segmentation_path = os.path.join(self.args.parent_dir , self.args.results_path.replace('results','segmentations'))
            if not os.path.exists(segmentation_path):
                os.mkdir(segmentation_path)
            
            mode = 'val'
            print(f'performance at epoch {epoch}')
            
            print("saving segmentation before viterbi")
            if self.args.save_seg  :

                colors = {}
                cmap = plt.get_cmap('tab20')
                for label_idx, label in enumerate(np.unique(gt_list)):
                    if label == -1:
                        
                        colors[label] = (0, 0, 0)
                    else:
                        # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                        colors[label] = cmap(label_idx / len(np.unique(gt_list)))
                
                for idx, video in tqdm(enumerate(video_names), position=0, leave=True):
                    
                    path = os.path.join(segmentation_path, video[0].split('/')[-1] + '.png')
                    plot_segm(path, segmentation_map_list[idx], colors, name=video[0].split('/')[-1])
            dir_name = os.path.join(self.args.results_dir, "training")
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            
            write_metrics_mof_f1(acc_cur ,return_stat['mean_f1'][0] , epoch, os.path.join(dir_name, "metrics_uas.md"))
            
            
            print('---------------')
            print(" After VITERBI : ") 
            accuracy  = Accuracy(self.args)
            accuracy.gt_labels = gt_list
            accuracy.predicted_labels = viterbi_label_list
            if self.args.bg:
                # enforce bg class to be bg class
                accuracy.exclude[-1] = [-1]
            old_mof, total_fr = accuracy.mof(old_gt2label=gt2label)
            gt2label = accuracy._gt2cluster
            f1_score = F1Score(K=num_classes, n_videos=len(testloader))
            
            label2gt = {}
            for key, val in gt2label.items():
                try:
                    label2gt[val[0]] = key
                except IndexError:
                    pass
            acc_cur = accuracy.mof_val()
            print(f"MoF val: {acc_cur}")
            print(f"previous dic -> MoF val: {float(old_mof) / total_fr}")
            average_class_mof = accuracy.mof_classes()
            return_stat = accuracy.stat()       
            print("MOF STATS: " , return_stat)
            f1_score.set_gt(gt_list)
            f1_score.set_pr(viterbi_label_list)
            f1_score.set_gt2pr(gt2label)
            
            if self.args.bg: 
                f1_score.set_exclude(-1) 
            f1_score.f1()
            for key, val in f1_score.stat().items():
                print("key :  " , key , "val : " , val)
                return_stat[key] = val
            write_return_stats(return_stat , epoch, os.path.join(dir_name, "return_stats.md"))    

            start_idx = 0
            for i in range(len(frame_count_list)):
                segmentation_map_list[i]['cl'] = (viterbi_label_list[start_idx : start_idx+frame_count_list[i]] , label2gt)
                start_idx = start_idx + frame_count_list[i]

            
            if self.args.save_seg  :
                segmentation_path = os.path.join(self.args.parent_dir , self.args.results_path.replace('results','segmentations_viterbi_uas'))
                if not os.path.exists(segmentation_path):
                    os.mkdir(segmentation_path)
                colors = {}
                cmap = plt.get_cmap('tab20')
                for label_idx, label in enumerate(np.unique(gt_list)):
                    if label == -1:
                        colors[label] = (0, 0, 0)
                    else:
                        colors[label] = cmap(label_idx / len(np.unique(gt_list)))

                
                for idx, video in tqdm(enumerate(video_names), position=0, leave=True):
                    vid_name = video[0].split('/')[-1]
                    pred_path = self.args.predictions_dir + '/'+ vid_name
                    path = os.path.join(segmentation_path, video[0].split('/')[-1] + '.png')
                   
                    labels,label2gt = segmentation_map_list[idx]['cl']
                    segm = list(map(lambda x: label2gt[int(x)], labels))
                    segm = [segm[i] if segm[i]!=-1 else -1 for i in range(len(segm))]
                    
                    with open(pred_path, 'w') as file:
                        for label in segm:
                            file.write(str(int(label))+ '\n')
                    plot_segm(path, segmentation_map_list[idx], colors, name=video[0].split('/')[-1])
            acc, edit, f1s = metrics_seg_viterbi.print(mode + " viterbi", epoch=epoch)
            dir_name = os.path.join(self.args.results_dir, "viterbi")
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            

            
            if eval_data is not None: 
                eval_data[epoch] = {
                    "mof": acc_cur,
                    "mean_f1" : return_stat['mean_f1'][0]
                }
            if not self.args.inference_only: 
                data = []
                for g, vals in eval_data.items():
                    for k, v in vals.items():
                        data.append([g, k, v])


                if self.args.log_wandb: 
                    
                    table = wandb.Table(data=data, columns = ["steps", "key", "value"])
                    wandb.log({"Per-Step Eval Data" : table})

                if is_final: 
                    max_step = 0
                    avg_max = 0 
                    for step in eval_data:
                        temp = (eval_data[step]["mof"] + eval_data[step]["mean_f1"]) / 2
                        if temp > avg_max:
                            avg_max = temp
                            max_step = step
                    data = [ ["epoch", max_step], 
                        ["mof", eval_data[max_step]["mof"]],  ["mean_f1", eval_data[max_step]["mean_f1"]]]

                    print("************************************")
                    print(f"Best Scores on Epoch #: {max_step}")
                    print(f"MoF: {eval_data[max_step]['mof']}")
                    print(f"Mean F1: {eval_data[max_step]['mean_f1']}")
                    print("************************************")
                    if self.args.log_wandb:
                        table = wandb.Table(data=data, columns=["type", "value"])
                        wandb.log(
                            {
                                "Evaluation Scores": table
                            }
                        )
                write_metrics_mof_f1(acc_cur ,return_stat['mean_f1'][0] , epoch, os.path.join(dir_name, "metrics_uas.md"))
                write_metrics(acc, edit, f1s, epoch, os.path.join(dir_name, "metrics.md"))
        return eval_data
                
            
               
    def convert_id_to_actions(self, framewise_predictions, gt_org, actions_dict_inv):
   
        recog = []
        for i in range(framewise_predictions.shape[1]):
            recog.extend([actions_dict_inv[framewise_predictions[:, i].item()]] * self.args.sample_rate)
      
        # adjust length of recog if there is a size mismatch with the ground truth
        if gt_org.shape[1] != len(recog):
            if gt_org.shape[1] < len(recog):
                recog = recog[:gt_org.shape[1]]
            elif gt_org.shape[1] > len(recog):
                recog = recog + recog[::-1]
                recog = recog[:gt_org.shape[1]]
        return recog
    
    
    def convert_segments_to_labels(self, action, duration, num_frames):
        assert  action.shape[0] == 1
        labels = action[0, :] - 2
        duration = duration[0, :]
        duration = duration / duration.sum()
        duration = (duration * num_frames).round().long()
        if duration.shape[0] == 0:
            duration = torch.tensor([num_frames])
            labels = torch.tensor([0])
        if duration.sum().item() != num_frames:
            # there may be small inconsistencies due to rounding.
            duration[-1] = num_frames - duration[:-1].sum()
        assert duration.sum().item() == num_frames, f"Prediction {duration.sum().item()} does not match number of frames {num_frames}."
        frame_wise_predictions = torch.zeros((1, num_frames))
        idx = 0
        for i in range(labels.shape[0]):
            frame_wise_predictions[0, idx:idx + duration[i]] = labels[i]
            idx += duration[i]
        return frame_wise_predictions


    def convert_labels_to_segments(self, labels):
        segments = self.convert_labels(labels)
        # we need to insert <sos> and <eos>
        segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
        segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
        
        target_labels = torch.stack([one_seg[0] for one_seg in segments]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
        
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).to(target_labels.device).unsqueeze(0)
        segments_dict = {'seg_gt': target_labels,
                        'seg_dur': target_durations_unnormalized,
                        'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                        }
        return segments_dict
        
    def convert_labels_to_segments_dur(self, labels): # , split_segments=False, split_segments_max_dur=None
        segments = self.convert_labels(labels)
        # we need to insert <sos> and <eos>
        segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
        segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
        if self.args.split_segments and   self.mode == 'train' and self.args.split_segments_max_dur: 
            max_dur = self.args.split_segments_max_dur # it used to be random.sample(split_segments_max_dur, 1)[0]
            segments = self.split_segments_into_chunks(segments, labels.shape[0], max_dur)
            
            
        target_labels = torch.stack([one_seg[0] for one_seg in segments]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
        
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).to(target_labels.device).unsqueeze(0)
        segments_dict = {'seg_gt': target_labels,
                        'seg_dur': target_durations_unnormalized,
                        'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                        }
        return segments_dict


    def compute_offsets(seldf, time_stamps):
        time_stamps.insert(0, -1)
        time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
        return time_stamps_unnormalized

    def convert_labels(self,labels):
        action_borders = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
        action_borders.insert(0, -1)
        action_borders.append(len(labels) - 1)
        label_start_end = []
        for i in range(1, len(action_borders)):
            label, start, end = labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
            label_start_end.append((label, start, end))
        return label_start_end


    def start_end2center_width(self, start_end):
        return torch.stack([start_end.mean(dim=2), start_end[:, :, 1] - start_end[:, :, 0]], dim=2)


    def convert_segments(self, segments):
        labels = np.zeros(segments[-1][-1] + 1)
        for segment in segments:
            labels[segment[1]:segment[2] + 1] = segment[0]
        return labels    
