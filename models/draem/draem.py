from torch import torch
from models.model_unet.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, ReconstructiveSubNetworkVAE, ReconstructiveSubNetworkVAEAttention, ReconstructiveSubNetworkAttention
from models.loss.loss import MSSIM, FocalLoss, SSIM, LpipsLoss
import lightning as L
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os

class DraemModel(L.LightningModule):
    def __init__(self, config: dict={}):
        self.config = config.copy()
        default_config = {
            "device": "cuda",
            "reconstructive_network_name": "Simple-Encoder",
            "reconstruction_loss": "SSIM",
            "learning_rate": 0.0001,
            "epochs": 700,
            "checkpoint_path": os.path.join(os.getcwd(), "checkpoints"),
            "load_checkpoints": False,
        }
        for key in default_config:
            if key not in self.config:
                self.config[key] = default_config[key]

        super(DraemModel, self).__init__()
        self.test_image_dimensions = 256
        self.current_epochs = 0
        super().__init__()

        if self.config["load_checkpoints"]:
            result = [None, None]
            with os.scandir(self.config["checkpoint_path"]) as entries:
                for entry in entries:
                    if entry.name.endswith("_seg.pckl"):
                        result[1] = entry.path
                    elif entry.name.endswith(".pckl"):
                        result[0] = entry.path
            if "VAE-Encoder" in result[0]:
                self.model =  ReconstructiveSubNetworkVAE(in_channels=3, out_channels=3, base_width=128)
            elif "Attention-Encoder" in result[0]:
                self.model = ReconstructiveSubNetworkAttention(in_channels=3, out_channels=3)
            elif "VAE-Attention-Encoder" in result[0]:
                self.model = ReconstructiveSubNetworkVAEAttention(in_channels=3, out_channels=3)
            else:
                self.model =  ReconstructiveSubNetwork(in_channels=3, out_channels=3)
            self.model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
            self.model.load_state_dict(torch.load(f'{result[0]}'))
            self.model_seg.load_state_dict(torch.load(f'{result[1]}'))
        else:
            self.l2 = torch.nn.modules.loss.MSELoss()
            if self.config["reconstruction_loss"] == "LPIPS":
                self.reconstruction_loss = LpipsLoss(device=self.config["device"])
            elif self.config["reconstruction_loss"] == "MSSIM":
                self.reconstruction_loss = MSSIM(device=self.config["device"])
            else:
                self.reconstruction_loss = SSIM(device=self.config["device"])
            self.focal_loss = FocalLoss()

            if self.config["reconstructive_network_name"] == "VAE-Encoder":
                self.model =  ReconstructiveSubNetworkVAE(in_channels=3, out_channels=3, base_width=128)
            elif self.config["reconstructive_network_name"] == "Attention-Encoder":
                self.model = ReconstructiveSubNetworkAttention(in_channels=3, out_channels=3)
            elif self.config["reconstructive_network_name"] == "VAE-Attention-Encoder":
                self.model = ReconstructiveSubNetworkVAEAttention(in_channels=3, out_channels=3)
            else:
                self.model =  ReconstructiveSubNetwork(in_channels=3, out_channels=3)
            self.model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        self.total_pixel_scores = []
        self.total_gt_pixel_scores = []
        self.anomaly_score_gt = []
        self.anomaly_score_prediction = []
        self.mask_count=0
        self.automatic_optimization=False
        self.test_step_outputs=[]

    def configure_optimizers(self):
            optimizer = torch.optim.Adam([
                                      {"params": self.model.parameters(), "lr": self.config["learning_rate"]},
                                      {"params": self.model_seg.parameters(), "lr": self.config["learning_rate"]}])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config["epochs"]*0.8,self.config["epochs"]*0.9],gamma=0.2, last_epoch=-1)
            return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
            }

    def forward(self, x):
        return self.shared_step(x)

    def shared_step(self, inputs):
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
        recon_loss = self.reconstruction_loss(preds["gray_rec"], gray_batch)
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
        self.log("train_loss", loss_sum, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_epoch", loss_sum, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx % 20 == 0:
            for idx, _ in enumerate(gray_batch):
                im = gray_batch[idx]
                gt_mask = anomaly_mask[idx]
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(0, batch_idx, idx), im)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/t_mask".format(0, batch_idx, idx), gt_mask)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/out_mask".format(0, batch_idx, idx), preds["out_mask_sm"][idx])
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/batch_augmented".format(0, batch_idx, idx), preds["aug_gray_batch"][idx])
        return {"loss": loss_sum}

    def on_train_epoch_end(self):
        self.current_epochs+=1
        prefix = os.path.join(
        self.config["checkpoint_path"],
        f"{self.config['reconstructive_network_name']}-"
        f"{self.config['reconstruction_loss']}-"
        f"{self.config['learning_rate']}-"
        f"{self.config['epochs']}"
        )
        if self.current_epochs in {1, round(self.config["epochs"]*0.25),
                                   round(self.config["epochs"]*0.5),
                                   round(self.config["epochs"]*0.75),
                                   self.config["epochs"]}:

            torch.save(self.model.state_dict(), os.path.join(prefix, "model.pckl"))
            torch.save(self.model_seg.state_dict(), os.path.join(prefix, "model_seg.pckl"))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        while batch_idx <= 4:
            gray_batch = batch["image"]
            gray_rec = self.model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
            out_mask = self.model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            t_mask = batch["anomaly_mask"]
            for idx, _ in enumerate(batch["image"]):
                im = batch["image"][idx]
                mask = out_mask_sm[idx]
                gt_mask = t_mask[idx]
                t_ch = torch.unsqueeze(torch.zeros_like(mask[0]), 0)
                heatmap = torch.cat((torch.unsqueeze(mask[1], 0), t_ch, torch.unsqueeze(mask[0], 0)))
                img = TF.to_pil_image(im)
                h_img = TF.to_pil_image(heatmap)
                res = Image.blend(img, h_img, 0.5)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(dataloader_idx, batch_idx, idx), im)
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/out_mask_sm_".format(dataloader_idx, batch_idx, idx), pil_to_tensor(res))
                self.logger.experiment.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/t_mask".format(dataloader_idx, batch_idx, idx), gt_mask)

    def test_step(self, batch, batch_idx):
        gray_batch = batch["image"]
        is_normal = batch["has_anomaly"].detach().cpu().numpy()[0 ,0]
        self.anomaly_score_gt.append(is_normal)
        true_mask = batch["mask"]
        true_mask_cv = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))
        gray_rec = self.model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = self.model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        for idx, _ in enumerate(batch["image"]):
            im = batch["image"][idx]
            recons = gray_rec[idx]
            self.test_image_dimensions = im.shape[-1]

            img = TF.to_pil_image(im)
            mask = out_mask_sm[idx]
            t_ch = torch.unsqueeze(torch.zeros_like(mask[0]), 0)

            heatmap = torch.cat((torch.unsqueeze(mask[1], 0), t_ch, torch.unsqueeze(mask[0], 0)))

            h_img = TF.to_pil_image(heatmap)

            res = Image.blend(img, h_img, 0.5)
            self.logger.experiment.add_image("batch_idx_{}_sample_idx_{}/image_".format(batch_idx, idx), im)
            self.logger.experiment.add_image("batch_idx_{}_sample_idx_{}/recons_".format(batch_idx, idx), recons)
            self.logger.experiment.add_image("batch_idx_{}_sample_idx_{}/heat_map_".format(batch_idx, idx), heatmap)
            self.logger.experiment.add_image("batch_idx_{}_sample_idx_{}/out_mask_sm_".format(batch_idx, idx), pil_to_tensor(res))
            self.logger.experiment.add_image("batch_idx_{}_sample_idx_{}/t_mask".format(batch_idx, idx), true_mask[idx])

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        self.anomaly_score_prediction.append(image_score)
        self.mask_count+=1
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        self.total_pixel_scores.extend(flat_out_mask)
        self.total_gt_pixel_scores.extend(flat_true_mask)

        return self.anomaly_score_prediction, self.anomaly_score_gt

    def on_test_epoch_end(self):
        self.anomaly_score_prediction = np.array(self.anomaly_score_prediction)
        self.anomaly_score_gt = np.array(self.anomaly_score_gt)
        self.total_gt_pixel_scores = np.array(self.total_gt_pixel_scores)
        auroc = roc_auc_score(self.anomaly_score_gt, self.anomaly_score_prediction)
        ap = average_precision_score(self.anomaly_score_gt, self.anomaly_score_prediction)

        self.total_gt_pixel_scores = self.total_gt_pixel_scores.astype(np.uint8)
        self.total_gt_pixel_scores = self.total_gt_pixel_scores[:self.test_image_dimensions * self.test_image_dimensions * self.mask_count]
        self.total_pixel_scores = self.total_pixel_scores[:self.test_image_dimensions * self.test_image_dimensions * self.mask_count]
        auroc_pixel = roc_auc_score(self.total_gt_pixel_scores, self.total_pixel_scores)
        ap_pixel = average_precision_score(self.total_gt_pixel_scores, self.total_pixel_scores)

        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print(f"Mask Counts: {self.mask_count}, prediction_anomaly_score_len: {len(self.anomaly_score_prediction)}), anomaly_score_gt: {len(self.anomaly_score_gt)}, total_gt_pixel_scores:{len(self.total_gt_pixel_scores)}")

        self.total_pixel_scores = []
        self.total_gt_pixel_scores = []
        self.anomaly_score_gt = []
        self.anomaly_score_prediction = []
        self.mask_count=0
        return {"auroc": auroc, "ap": ap}

    def on_load_checkpoint(self, model, model_seg):
        torch.load_from_checkpoint(self.model.state_dict(), model)
        torch.load_from_checkpoint(self.model_seg.state_dict(), model_seg)
