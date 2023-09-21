#######################################
# code written by S. Alireza Golestaneh
#######################################
import copy
import random
from ast import arg
from collections import defaultdict
from encoder_mlp import MLP
from tot import generate_optimal_transport, get_complete_cost_matrix
import numpy as np 
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from bmvc_advanced import Encoder_advanced
from bmvc_enc_dec import Decoder_org_enc_dec, Encoder_org_enc_dec
from bmvc_org import Encoder_org
from PE import SinusoidalPositionalEmbedding
from segment_embedding import SegmentEmbedding
from utils import exponential_descrease, remove_duplicates_from_transcript 
from uvast_dec import TransformerDecoder_UVAST, TransformerDecoderLayer_UVAST



class encoder_asformer_org_enc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc = Encoder_org(num_layers=args.num_layers_enc, r1=2, r2=2, num_f_maps=64, input_dim=args.features_dim, num_classes=args.num_classes, channel_masking_rate=args.channel_masking_rate, att_type='sliding_att', alpha=1, device=args.device)
        
    def forward(self, inputs, masks):
        outputs = []
        cls_framewise, latentfeat_framewise = self.enc(inputs, masks)
        outputs.append(cls_framewise)
        return outputs, latentfeat_framewise
        
        
class encoder_asformer_advanced_enc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc = Encoder_advanced(num_layers=args.num_layers_enc, r1=2, r2=2, num_f_maps=args.num_f_maps, input_dim=args.features_dim, num_classes=args.num_classes, channel_masking_rate=args.channel_masking_rate, att_type='sliding_att', args=args)
        
    def forward(self, inputs, masks):
        outputs = []
        cls_framewise, latentfeat_framewise = self.enc(inputs, masks, istraining=True)
        outputs.append(cls_framewise)
        return outputs, latentfeat_framewise


class encoder_asformer_org_enc_dec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc = Encoder_org_enc_dec(num_layers=args.num_layers_enc, r1=2, r2=2, num_f_maps=64, input_dim=args.features_dim, num_classes=args.num_classes, channel_masking_rate=args.channel_masking_rate, att_type='sliding_att', alpha=1, device=args.device)
        self.dec = nn.ModuleList([copy.deepcopy(Decoder_org_enc_dec(num_layers=args.num_layers_asformer_dec, r1=2, r2=2, num_f_maps=64, input_dim=args.num_classes, num_classes=args.num_classes, att_type='sliding_att', alpha=exponential_descrease(s), device=args.device)) for s in range(args.num_layers_asformer_dec_repeat)])
        
    def forward(self, inputs, masks):
        outputs = []
        out, feature = self.enc(inputs, masks)
        outputs.append(out)
        for decoder in self.dec:
            out, feature = decoder(nn.functional.softmax(out, dim=1) * masks[:, 0:1, :], feature* masks[:, 0:1, :], masks)
            outputs.append(out)

        return outputs, feature


class decoder_duration(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if self.args.alignment_decoder_model == 'uvast_decoder':
            self.decoder_duration = TransformerDecoder_UVAST(TransformerDecoderLayer_UVAST(d_model=args.num_f_maps, nhead=args.n_head_dec_dur_uvast, activation=args.activation, 
                                                                                           dropout=args.dropout_dec_dur, dim_feedforward=args.dec_dim_feedforward_dec_dur, args=args), 
                                                                                           repeat_mod=1, num_layers=args.num_layers_dec_dur, args=args)
        elif self.args.alignment_decoder_model == 'pytorch_decoder':
            self.decoder_duration = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.num_f_maps, 
                                                                                     nhead=args.n_head_dec_dur_pytorch,
                                                                                     activation=args.activation,  
                                                                                     dropout=args.dropout_dec_dur,
                                                                                     dim_feedforward=args.dec_dim_feedforward_dec_dur), 
                                                                                     num_layers=args.num_layers_trf_dec_dur_pytorch)

        self.pos_embed_dur2 = SinusoidalPositionalEmbedding(embedding_dim=args.num_f_maps, padding_idx=0, init_size=2 * args.num_classes)

        self.dropout_dur1 = nn.Dropout(p=0.2)
        self.dropout_dur2 = nn.Dropout(p=0.2)
    
    # args
    #   - enc_feat: framewise features of the encoder model
    #   - dec_feat: segmentwise features of the decoder model
    #   - pred_transcript: predicted transcript of the decoder model
    #   - gt_transcript: ground truth transcript -> only used for training
    def forward(self, enc_feat, dec_feat, pred_transcript=None, no_split_data=None, gt_transcript=None, mode= 'train'):         
        tgt_pe_dur = torch.tensor([0]).to(self.args.device)
        tgt_pe = torch.tensor([0]).to(self.args.device)
      
        if no_split_data is not None:
            # during training we need to remove duplicates from the transcript using the gt transcript
           
         
           
            new_feat_seg= dec_feat 
            pred_seg_cls_ids_refine = pred_transcript

            dec_feat_refined = einops.rearrange(new_feat_seg, 'B S E  ->  S B E')  
            dec_feat_refined = dec_feat
           
        else:
           
            if mode =='train':
                dec_feat_refined= dec_feat
            elif mode =='test':

                dec_feat_refined = einops.rearrange(dec_feat, 'B S E  ->  S B E')
            
        # rearrange and add positional encoding
        enc_feat = einops.rearrange(enc_feat, 'B E S ->  S B E')
        dec_feat_refined = einops.rearrange(dec_feat_refined, 'S B E -> S B E')  
        tgt_pe = self.pos_embed_dur2(torch.ones(1, dec_feat_refined.shape[0]).to(self.args.device))
        tgt_pe_dur = einops.rearrange(tgt_pe, 'B T E -> T B E') * self.args.add_tgt_pe_dec_dur
       
        # align framewise encoder features to segmentwise decoder features 
        x = self.dropout_dur1(enc_feat)
        y = self.dropout_dur2(dec_feat_refined) + tgt_pe_dur.clone()
       
        aligned_feat = self.decoder_duration(x, y)
        if self.args.alignment_decoder_model == 'uvast_decoder':
            aligned_feat = aligned_feat[0][-1] # this decoder returns a list                   
                    
        aligned_encoder_feat = einops.rearrange(aligned_feat, 'B T E -> T E B')
        dec_feat_refined = einops.rearrange(dec_feat_refined, 'B T E -> T B E')
        
        frames_to_segment_assignment = torch.bmm(dec_feat_refined + tgt_pe, aligned_encoder_feat)
        frames_to_segment_assignment = einops.rearrange(frames_to_segment_assignment, 'B S T -> B T S')

        return frames_to_segment_assignment

    def remove_duplicates(self, dec_feat, pred_transcript, gt_transcript, no_split_data):
        # pred_seg_cls is the output of decoder for seg classes
        pred_seg_cls_ids = torch.max(pred_transcript, 1)[1]
     
        pred_seg_cls_ids[gt_transcript == 1] = -1
        # pred_seg_cls_ids has splits, but we dont want splits, so we will refine it so there is no split
      
        pred_seg_cls_ids_refine = []
        pj = 1
        for pi in range(len(pred_seg_cls_ids[0])):
            if pj < len(pred_seg_cls_ids[0]):
                if pred_seg_cls_ids[0][pi] != pred_seg_cls_ids[0][pj]:
                    pred_seg_cls_ids_refine.append(pred_seg_cls_ids[0][pi].item())
                    pj += 1
                elif pred_seg_cls_ids[0][pi] == pred_seg_cls_ids[0][pj]:
                    pj += 1
            else:
                if pred_seg_cls_ids[0][pi] != pred_seg_cls_ids_refine[-1]:
                    pred_seg_cls_ids_refine.append(pred_seg_cls_ids[0][pi].item())
        if -1 in pred_seg_cls_ids_refine:
            pred_seg_cls_ids_refine.pop(pred_seg_cls_ids_refine.index(-1))
        
        dict_clsid_feat = defaultdict(list)
        for kkind, valll in enumerate(pred_seg_cls_ids[0]):
            dict_clsid_feat[valll.item()].append(dec_feat[kkind])
        dict_clsid_feat.pop(-1, None)
    
        seg_gt_no_split, seg_dur_no_split = no_split_data[0], no_split_data[1]
        New_feat_seg = torch.zeros(1, seg_gt_no_split.shape[1], self.args.num_f_maps).to(self.args.device)

        for i in range(seg_gt_no_split.shape[1]):
            if dict_clsid_feat.get(seg_gt_no_split[:, i].item()) is not None:
                if len(dict_clsid_feat[seg_gt_no_split[:, i].item()]) == 1:
                    New_feat_seg[:, i] = dict_clsid_feat[seg_gt_no_split[:, i].item()][0]
                else:
                    New_feat_seg[:, i] = dict_clsid_feat[seg_gt_no_split[:, i].item()][random.randint(0, len(dict_clsid_feat[seg_gt_no_split[:, i].item()]) - 1)]
            else:
                if i < len(pred_seg_cls_ids_refine):
                    New_feat_seg[:, i] = dict_clsid_feat[pred_seg_cls_ids_refine[i]][0]
                else:
                    rnd_key = list(dict_clsid_feat.keys())[random.randint(0, len(list(dict_clsid_feat.keys())) - 1)]
                    if len(dict_clsid_feat[rnd_key]) == 1:
                        New_feat_seg[:, i] = dict_clsid_feat[rnd_key][0]
                    else:
                        New_feat_seg[:, i] = dict_clsid_feat[rnd_key][random.randint(0, len(dict_clsid_feat[rnd_key]) - 1)]
        return pred_seg_cls_ids_refine, New_feat_seg

class uvast_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # initialize encoder model
        if self.args.encoder_model == 'asformer_org_enc':
            self.enc_feat = encoder_asformer_org_enc(args)
        elif self.args.encoder_model == 'asformer_advanced':
            self.enc_feat = encoder_asformer_advanced_enc(args)
        elif self.args.encoder_model == 'asformer_org_enc_dec':
            self.enc_feat = encoder_asformer_org_enc_dec(args)
        elif self.args.encoder_model == 'mlp':
            self.enc_feat  = MLP(args)
        # initialize transcript decoder model
        
        self.dec_action = TransformerDecoder_UVAST(TransformerDecoderLayer_UVAST(d_model=args.num_f_maps, 
                                                                                 nhead=args.n_head_dec, 
                                                                                 activation=args.activation, 
                                                                                 dropout=args.dropout, 
                                                                                 dim_feedforward=args.dec_dim_feedforward, args=args), 
                                                   repeat_mod=1, num_layers=args.num_layers_dec, args=args)

        # set up positional encoding and segment embedding
        if args.use_pe_tgt or args.use_pe_src:
            self.pos_embed = SinusoidalPositionalEmbedding(embedding_dim=args.num_f_maps, padding_idx=0, init_size=2 * args.num_classes)
        self.dec_embedding = SegmentEmbedding(args)       
            
        # initialize alignment decoder for duration prediction
        self.prediction_action = nn.Linear(args.num_f_maps, args.num_classes + 2, bias=False)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_action = nn.Dropout(p=0.5)
        
        # if self.args.use_alignment_dec:
        #     self.dec_duration = decoder_duration(args)

        self._init_weights()

    def forward(self, inputs, mask, seg_data=None, attn_mask_gt=None, no_split_data=None, mode='train'):
        frames_to_segment_assignment = 0.0
        pred_transcripts = 0.0
        pred_crossattn = 0.0
        pred_framewise, feat_enc = self.enc_feat(inputs, mask)
       
        pred_transcript_AD = None
        pred_dur_AD = None
        if  mode=='train': 
            if self.args.use_transcript_dec:
                p_gauss = get_complete_cost_matrix(vid_len= inputs.shape[2] , num_clusters=self.args.num_classes , num_videos=self.args.num_videos , sigma=self.args.sigma)
                
                q = generate_optimal_transport(torch.permute(pred_framewise[0][0], (1,0)), self.args.epsilon, p_gauss)
                clusters_idx= np.argmax(q.clone().detach().cpu().numpy(), axis= 0)
                sorted_idx= sorted(clusters_idx)
                segment_order, transcript= [] , []
                for id in sorted_idx: 
                    segment_order.append(np.where(clusters_idx==id)[0][0]+2) #added 2 for sos eos order
                    transcript.append(np.where(clusters_idx==id)[0][0])
               
                
                if self.args.fixorder_T: 
                    #for fix order
                    segment_order=[x+2 for x in range(self.args.num_classes)]
                segment_order.insert(0, 0.) #sos
                segment_order.append(-1.)   #eos
                segment_len= len(segment_order)
                segment_order= torch.Tensor([segment_order]).to(inputs.device).to(torch.float32)
                non_list= [None for i in range(segment_len)]
                seg_data = (segment_order ,non_list)
                tgt_emb_clsids, tgt_mask_from_pad = self.dec_embedding(seg_data)
            
                tgt_mask, tgt_pe, src_pe, src_key_padding_mask, tgt_key_padding_mask = self.generate_pe_and_masks(tgt_emb_clsids, tgt_mask_from_pad, feat_enc, mask)
                src = feat_enc.clone()
                src = einops.rearrange(src, 'B E S -> S B E')
                tgt = einops.rearrange(tgt_emb_clsids, 'B T E -> T B E')
                tgt = self.dropout_1(tgt)
                src = self.dropout_2(src)
                decoder_output, _, pred_crossattn, _ = self.dec_action(tgt=tgt + tgt_pe, 
                                                                    memory=src + src_pe, 
                                                                    tgt_mask=tgt_mask, 
                                                                    memory_key_padding_mask=src_key_padding_mask, 
                                                                    tgt_key_padding_mask=tgt_key_padding_mask)

                
                pred_transcripts = [] 
                for iii in range(len(decoder_output)):
                    out_dec = einops.rearrange(decoder_output[iii], 'T B E -> B T E')
                    out_dec = self.dropout_action(out_dec)
                    pred_transcript = einops.rearrange(self.prediction_action(out_dec), 'B T E -> B E T')
                    pred_transcripts.append(pred_transcript)
            
           
            return pred_framewise, pred_transcripts, pred_crossattn, frames_to_segment_assignment 
       
        
        if self.args.use_transcript_dec and mode=='test':
            # <sos> token as initialization
            seq = torch.tensor([[0]]).to(inputs.device)
            dur = torch.tensor([[0.0]]).to(inputs.device)
            # start predicting the seq

            while seq[0, -1].item() != 1 and len(seq[0, :]) < self.args.len_seg_max:
                tgt_emb_clsids, tgt_mask_from_pad = self.dec_embedding((seq, dur))
                tgt_mask, tgt_pe, src_pe, _, _ = self.generate_pe_and_masks(tgt_emb_clsids, tgt_mask_from_pad, feat_enc, mask)
                src = einops.rearrange(feat_enc, 'B E S -> S B E')
                tgt = einops.rearrange(tgt_emb_clsids, 'B T E -> T B E')
                
                # transcript decoder
                decoder_output, _, pred_crossattn, _ = self.dec_action(tgt=tgt + tgt_pe,
                 memory=src + src_pe.clone(), tgt_mask=tgt_mask)
                
                out_dec = einops.rearrange(decoder_output[-1], 'T B E -> B T E')
                pred_action = self.prediction_action(out_dec * tgt_mask_from_pad.unsqueeze(-1))
                _ , pred_action = torch.max(pred_action.data, 2)
                
                seq = torch.cat([seq, pred_action[:, -1:]], dim=1)

            seq = seq[:, 1:-1]
            pred_transcript = seq.clone()
            if pred_crossattn is not None:
                dur = torch.softmax(pred_crossattn[0] / 0.001, dim=1).sum(2) / torch.softmax(pred_crossattn[0] / 0.001, dim=1).sum()
            else:
                dur = torch.tensor([1 / seq.shape[1]] * (seq.shape[1] + 1)).to(seq.device).unsqueeze(0)
            
            

            pred_transcript_AD = None
            pred_dur_AD = None
            
            assert seq.shape == dur[:, 1:].shape
            return pred_framewise, seq, dur[:, 1:], pred_dur_AD, pred_transcript_AD
            #return pred_framewise, seq, pred_crossattn, pred_dur_AD, pred_transcript_AD
        #print(' cross attnetion :', pred_crossattn[0].shape)
        return pred_framewise, pred_transcripts, pred_crossattn, pred_dur_AD, pred_transcript_AD

    
    def generate_pe_and_masks(self, tgt_emb_clsids, tgt_mask_from_pad, feat_enc, mask):
        #print("PE MASK ")
        #print(tgt_emb_clsids.shape)
        
        tgt_mask = self.generate_square_subsequent_mask(int(tgt_emb_clsids.shape[1])).to('cuda')
        if self.args.use_pe_tgt:
            tgt_pos = self.pos_embed(torch.ones_like(tgt_emb_clsids)[:, :, 0]) * tgt_mask_from_pad.unsqueeze(-1)
            tgt_pe = einops.rearrange(tgt_pos, 'B T E -> T B E')
        else:
            tgt_pe = torch.tensor([0.0]).to('cuda')
        if self.args.use_pe_src:
            src_pos = (self.pos_embed(torch.ones_like(feat_enc.permute(0, 2, 1))[:, :, 0]) * (mask.permute(0, 2, 1)[:, :, 0:1]))
            src_pe = einops.rearrange(src_pos, 'B T E -> T B E')
        else:
            src_pe = src_pe = torch.tensor([0.0]).to('cuda')
        src_key_padding_mask = mask[:, 0, :].clone()
        tgt_key_padding_mask = tgt_mask_from_pad.clone()
        src_key_padding_mask = (1 - src_key_padding_mask).type(torch.BoolTensor).to('cuda')
        tgt_key_padding_mask = (1 - tgt_key_padding_mask).type(torch.BoolTensor).to('cuda')
        return tgt_mask, tgt_pe, src_pe, src_key_padding_mask, tgt_key_padding_mask


    def generate_square_subsequent_mask(self, sz: int) :
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def get_prototype(self):

        if self.args.encoder_model != 'mlp':
            return self.enc_feat.enc.conv_out.weight.data.clone()
        else:
            return self.enc_feat.get_prototypes()
    
    def convert_labels_to_segments(self, labels): # , split_segments=False, split_segments_max_dur=None
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
        
    def convert_labels(self,labels):
        action_borders = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
        action_borders.insert(0, -1)
        action_borders.append(len(labels) - 1)
        label_start_end = []
        for i in range(1, len(action_borders)):
            label, start, end = labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
            label_start_end.append((label, start, end))
        return label_start_end
    def compute_offsets(seldf, time_stamps):
        time_stamps.insert(0, -1)
        time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
        return time_stamps_unnormalized  

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
        
                module.weight.data.normal_(mean=0.0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()