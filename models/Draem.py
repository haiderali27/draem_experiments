from torch import torch
from models.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, ReconstructiveSubNetworkVAE, ReconstructiveSubNetworkVAEAttention, ReconstructiveSubNetworkAttention
from models.loss import MSSIM, FocalLoss, SSIM, LpipsLoss
import lightning as L
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
#from tensorboard_visualizer import TensorboardVisualizer

class DraemModel(L.LightningModule):
    def __init__(self, USE_MODEL='Default', recon_loss='ssim', lr=0.0001, epochs=700, load_check=False, load_check_models=[],checkpoint_path=None, checkpoint_prefix="", test_img_dim=256, visualize=True):
        super(DraemModel, self).__init__()
        self.img_dim=test_img_dim
        self.epochs = epochs
        self.learning_rate = lr
        self.current_epochs = 0
        self.recon_loss = recon_loss
        self.visualize=visualize
        
        self.checkpoint_path = checkpoint_path
     
        if checkpoint_path is None:
            self.checkpoint_path = os.path.join(os.getcwd(), 'checkpoints')
        
        self.checkpoint_prefix = f'{checkpoint_prefix}'

      
        super().__init__()
        if USE_MODEL=='VAE':
            self.model =  ReconstructiveSubNetworkVAE(in_channels=3, out_channels=3, base_width=128)
            print('VAE?')
        elif USE_MODEL=='Attention':
            self.model = ReconstructiveSubNetworkAttention(in_channels=3, out_channels=3)
            print('Attention?')
        elif USE_MODEL=='VAEAttention':
            self.model = ReconstructiveSubNetworkVAEAttention(in_channels=3, out_channels=3)
            print("VAE Attention")
        else:
            self.model =  ReconstructiveSubNetwork(in_channels=3, out_channels=3)
            print("Default")
        self.model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        self.l2 = torch.nn.modules.loss.MSELoss()
        self.ssim_loss = SSIM()
        self.mssim = MSSIM()
        self.lpips_loss= LpipsLoss()
        self.focal_loss = FocalLoss()
        self.total_pixel_scores = []
        self.total_gt_pixel_scores = []
        self.anomaly_score_gt = []
        self.anomaly_score_prediction = []
        self.msk_count=0
        self.automatic_optimization=False
        self.test_step_outputs=[]
        if load_check:
            self.model.load_state_dict(torch.load(load_check_models[0]))
            self.model_seg.load_state_dict(torch.load(load_check_models[1]))

        
    def forward(self, x):
        return self.shared_step(x)
    
    def shared_step(self, inputs):
        loss = dict()
        predictions = dict()
        aug_gray_batch = inputs["augmented_image"]

        gray_rec = self.model(aug_gray_batch)
        joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

        out_mask = self.model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        predictions.update({"out_mask_sm": out_mask_sm, "gray_rec": gray_rec, "aug_gray_batch":aug_gray_batch})
        return predictions
    
    def training_step(self, batch, batch_idx):
        batch_size = len(batch)

        opt = self.optimizers()
        
        preds = self.shared_step(batch)
        #loss
        gray_batch = batch["image"]
        anomaly_mask = batch["anomaly_mask"]

        l2_loss = self.l2(preds["gray_rec"],gray_batch)
        if self.recon_loss =='ssim':
            recon_loss = self.ssim_loss(preds["gray_rec"], gray_batch)
        elif self.recon_loss=='lpips':
            recon_loss = self.lpips_loss(preds["gray_rec"], gray_batch)
        elif self.recon_loss=='mssim':
            recon_loss = self.mssim(preds["gray_rec"], gray_batch)
        elif self.recon_loss=='combined':
            recon_loss1 = self.ssim_loss(preds["gray_rec"], gray_batch)
            recon_loss2 = self.lpips_loss(preds["gray_rec"], gray_batch)
            recon_loss = (recon_loss1+recon_loss2)

            
        segment_loss = self.focal_loss(preds["out_mask_sm"], anomaly_mask)
        loss_sum = l2_loss + recon_loss + segment_loss

        # Optimize
        opt.zero_grad()
        self.manual_backward(loss_sum)
        opt.step()

        loss_dict = dict()
        loss_dict.update({"l2_loss":l2_loss , "recon_loss":recon_loss , "segment_loss":segment_loss , "loss_sum":loss_sum })
        # Add losses to logs
        [self.log(k, v, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True) for k,v in loss_dict.items()]
        self.log('train_loss', loss_sum, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train_loss_epoch', loss_sum, on_step=False, on_epoch=True, sync_dist=True)
        
  
        if self.visualize and batch_idx % 400 == 0:
            #t_mask = preds["out_mask_sm"][:, 1:, :, :]
            for idx, _ in enumerate(gray_batch): 
                im = gray_batch[idx]
                gt_mask = anomaly_mask[idx]
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(0, batch_idx, idx), im)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/t_mask".format(0, batch_idx, idx), gt_mask)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/out_mask".format(0, batch_idx, idx), preds["out_mask_sm"][idx])
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/batch_augmented".format(0, batch_idx, idx), preds["aug_gray_batch"][idx])

            #self.logger.experiment.add_image("batch_augmented", preds["aug_gray_batch"])            
            #self.logger.experiment.add_image("batch_recon_target", gray_batch)
            #self.logger.experiment.add_image("batch_recon_out", preds["gray_rec"])
            #self.logger.experiment.add_image("mask_target", anomaly_mask)
            #self.logger.experiment.add_image("mask_out", t_mask)

        return {'loss': loss_sum}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        while batch_idx <= 4:
            gray_batch = batch["image"]
            gray_rec = self.model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
            

            out_mask = self.model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            t_mask = batch["anomaly_mask"]
            # print(gray_batch.shape, t_mask.shape)

            for idx, _ in enumerate(batch["image"]): 
                tb_logger = self.logger.experiment
                
                im = batch["image"][idx]
                mask = out_mask_sm[idx]
                gt_mask = t_mask[idx]
                t_ch = torch.unsqueeze(torch.zeros_like(mask[0]), 0)

                heatmap = torch.cat((torch.unsqueeze(mask[1], 0), t_ch, torch.unsqueeze(mask[0], 0)))
                img = TF.to_pil_image(im)  
                h_img = TF.to_pil_image(heatmap)

                res = Image.blend(img, h_img, 0.5)
                
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(dataloader_idx, batch_idx, idx), im)
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/out_mask_sm_".format(dataloader_idx, batch_idx, idx), pil_to_tensor(res))
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/t_mask".format(dataloader_idx, batch_idx, idx), gt_mask)
            break
    

    def test_step(self, batch, batch_idx):

        tb_logger = self.logger.experiment
        gray_batch = batch["image"]

        is_normal = batch["has_anomaly"].detach().cpu().numpy()[0 ,0]
        self.anomaly_score_gt.append(is_normal)
        true_mask = batch["mask"]
        
        true_mask_cv = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = self.model(gray_batch)


        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = self.model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        
        #print('#############', len(batch['image']),isinstance(batch['image'], Iterable))
        #if self.test_weights:
            #for index, key in enumerate(batch.keys()):
            #for idx, _ in enumerate(batch): 
            #    print(f"Index: {index}, key:{key}, batch_idx:{batch_idx}")
                #print(f"Index: {index}, Key: {key}, Value: {batch[key]}")


        for idx, _ in enumerate(batch["image"]):
           
            im = batch["image"][idx]
            recons = gray_rec[idx]
        
            img = TF.to_pil_image(im)  
            mask = out_mask_sm[idx]
            t_ch = torch.unsqueeze(torch.zeros_like(mask[0]), 0)

            heatmap = torch.cat((torch.unsqueeze(mask[1], 0), t_ch, torch.unsqueeze(mask[0], 0)))

            h_img = TF.to_pil_image(heatmap)

            res = Image.blend(img, h_img, 0.5)

            tb_logger.add_image("batch_idx_{}_sample_idx_{}/image_".format(batch_idx, idx), im)
            tb_logger.add_image("batch_idx_{}_sample_idx_{}/recons_".format(batch_idx, idx), recons)
            tb_logger.add_image("batch_idx_{}_sample_idx_{}/heat_map_".format(batch_idx, idx), heatmap)
            tb_logger.add_image("batch_idx_{}_sample_idx_{}/out_mask_sm_".format(batch_idx, idx), pil_to_tensor(res))
            tb_logger.add_image("batch_idx_{}_sample_idx_{}/t_mask".format(batch_idx, idx), true_mask[idx])


        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()

        image_score = np.max(out_mask_averaged)

        self.anomaly_score_prediction.append(image_score)
        self.msk_count+=1
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        self.total_pixel_scores.extend(flat_out_mask)
        self.total_gt_pixel_scores.extend(flat_true_mask)


        return self.anomaly_score_prediction, self.anomaly_score_gt

    def on_test_epoch_end(self):


        self.anomaly_score_prediction = np.array(self.anomaly_score_prediction)
        self.anomaly_score_gt = np.array(self.anomaly_score_gt)
        #self.total_gt_pixel_scores = np.array(self.total_gt_pixel_scores)
        self.total_gt_pixel_scores = np.array(self.total_gt_pixel_scores)
        auroc = roc_auc_score(self.anomaly_score_gt, self.anomaly_score_prediction)
        ap = average_precision_score(self.anomaly_score_gt, self.anomaly_score_prediction)

        self.total_gt_pixel_scores = self.total_gt_pixel_scores.astype(np.uint8)
        self.total_gt_pixel_scores = self.total_gt_pixel_scores[:self.img_dim * self.img_dim * self.msk_count]
        self.total_pixel_scores = self.total_pixel_scores[:self.img_dim * self.img_dim * self.msk_count]
        auroc_pixel = roc_auc_score(self.total_gt_pixel_scores, self.total_pixel_scores)
        ap_pixel = average_precision_score(self.total_gt_pixel_scores, self.total_pixel_scores)

        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print(f'Mask Counts: {self.msk_count}, prediction_anomaly_score_len: {len(self.anomaly_score_prediction)}), anomaly_score_gt: {len(self.anomaly_score_gt)}, total_gt_pixel_scores:{len(self.total_gt_pixel_scores)}')
        print("==============================")
        self.total_pixel_scores = []
        self.total_gt_pixel_scores = []
        self.anomaly_score_gt = []
        self.anomaly_score_prediction = []
        self.msk_count=0
        return {"auroc": auroc, "ap": ap}
    
    def on_load_checkpoint(self, model, model_seg):
        torch.load_from_checkpoint(self.model.state_dict(), model)
        torch.load_from_checkpoint(self.model_seg.state_dict(), model_seg)

    
    def on_train_epoch_end(self):
        self.current_epochs+=1
        if self.epochs==self.current_epochs or (self.current_epochs % 100 == 0):
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'{self.checkpoint_prefix}_{self.current_epochs}.pckl'))
            torch.save(self.model_seg.state_dict(), os.path.join(self.checkpoint_path, f'{self.checkpoint_prefix}_{self.current_epochs}_seg.pckl'))



    def configure_optimizers(self):
            
            #print("Optimizer - using {} with lr {}".format(self.cfg.SOLVER.NAME, self.cfg.SOLVER.BASE_LR))

            optimizer = torch.optim.Adam([
                                      {"params": self.model.parameters(), "lr": self.learning_rate},
                                      {"params": self.model_seg.parameters(), "lr": self.learning_rate}])

            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.epochs*0.8,self.epochs*0.9],gamma=0.2, last_epoch=-1)


            return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
            }
