import os
from glob import glob
from logging import getLogger
from typing import Literal, Optional, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import Dataset
from .pretrained import pretrained_checkpoints
from .constants import *
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm.auto import tqdm
from huggingface_hub import HfApi, upload_folder

from .synthesizer import commons
from .synthesizer.models import (
    SynthesizerTrnMs768NSFsid,
    MultiPeriodDiscriminator,
)

from .utils.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from .utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .utils.data_utils import TextAudioCollateMultiNSFsid

logger = getLogger(__name__)


class TrainingCheckpoint:
    def __init__(
        self,
        epoch: int,
        G: SynthesizerTrnMs768NSFsid,
        D: MultiPeriodDiscriminator,
        optimizer_G: torch.optim.AdamW,
        optimizer_D: torch.optim.AdamW,
        scheduler_G: torch.optim.lr_scheduler.ExponentialLR,
        scheduler_D: torch.optim.lr_scheduler.ExponentialLR,
        loss_gen: float,
        loss_fm: float,
        loss_mel: float,
        loss_kl: float,
        loss_gen_all: float,
        loss_disc: float,
    ):
        self.epoch = epoch
        self.G = G
        self.D = D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        self.loss_gen = loss_gen
        self.loss_fm = loss_fm
        self.loss_mel = loss_mel
        self.loss_kl = loss_kl
        self.loss_gen_all = loss_gen_all
        self.loss_disc = loss_disc

    def save(
        self,
        exp_dir="./",
        g_checkpoint: str | None = None,
        d_checkpoint: str | None = None,
    ):
        g_path = g_checkpoint if g_checkpoint is not None else f"G_latest.pth"
        d_path = d_checkpoint if d_checkpoint is not None else f"D_latest.pth"
        torch.save(
            {
                "epoch": self.epoch,
                "model": self.G.state_dict(),
                "optimizer": self.optimizer_G.state_dict(),
                "scheduler": self.scheduler_G.state_dict(),
                "loss_gen": self.loss_gen,
                "loss_fm": self.loss_fm,
                "loss_mel": self.loss_mel,
                "loss_kl": self.loss_kl,
                "loss_gen_all": self.loss_gen_all,
                "loss_disc": self.loss_disc,
            },
            os.path.join(exp_dir, g_path),
        )
        torch.save(
            {
                "epoch": self.epoch,
                "model": self.D.state_dict(),
                "optimizer": self.optimizer_D.state_dict(),
                "scheduler": self.scheduler_D.state_dict(),
            },
            os.path.join(exp_dir, d_path),
        )


def latest_checkpoint_file(files: list[str]) -> str:
    try:
        return max(files, key=lambda x: int(Path(x).stem.split("_")[1]))
    except:
        return max(files, key=os.path.getctime)


class RVCTrainer:
    def __init__(
        self,
        exp_dir: str,
        dataset_train: Dataset,
        dataset_test: Optional[Dataset] = None,
        sr: int = SR_48K,
    ):
        self.exp_dir = exp_dir
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.sr = sr
        self.writer = SummaryWriter(
            os.path.join(exp_dir, "logs", time.strftime("%Y%m%d-%H%M%S"))
        )

    def latest_checkpoint(self, fallback_to_pretrained: bool = True):
        files_g = glob(os.path.join(self.exp_dir, "G_*.pth"))
        if not files_g:
            return pretrained_checkpoints() if fallback_to_pretrained else None
        latest_g = latest_checkpoint_file(files_g)

        files_d = glob(os.path.join(self.exp_dir, "D_*.pth"))
        if not files_d:
            return pretrained_checkpoints() if fallback_to_pretrained else None
        latest_d = latest_checkpoint_file(files_d)

        return latest_g, latest_d

    def setup_models(
        self,
        resume_from: Tuple[str, str] | None = None,
        accelerator: Accelerator | None = None,
        lr=1e-4,
        lr_decay=0.999875,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps=1e-9,
        use_spectral_norm=False,
        segment_size=17280,
        filter_length=N_FFT,
        hop_length=HOP_LENGTH,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock: Literal["1", "2"] = "1",
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=512,
        upsample_rates: list[int] = [12, 10, 2, 2],
        upsample_kernel_sizes: list[int] = [24, 20, 4, 4],
        spk_embed_dim=109,
        gin_channels=256,
    ) -> Tuple[
        SynthesizerTrnMs768NSFsid,
        MultiPeriodDiscriminator,
        torch.optim.AdamW,
        torch.optim.AdamW,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.ExponentialLR,
        int,
    ]:
        if accelerator is None:
            accelerator = Accelerator()

        G = SynthesizerTrnMs768NSFsid(
            spec_channels=filter_length // 2 + 1,
            segment_size=segment_size // hop_length,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            spk_embed_dim=spk_embed_dim,
            gin_channels=gin_channels,
            sr=self.sr,
        ).to(accelerator.device)
        D = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm).to(
            accelerator.device
        )

        optimizer_G = torch.optim.AdamW(
            G.parameters(),
            lr,
            betas=betas,
            eps=eps,
        )
        optimizer_D = torch.optim.AdamW(
            D.parameters(),
            lr,
            betas=betas,
            eps=eps,
        )

        if resume_from is not None:
            g_checkpoint, d_checkpoint = resume_from
            logger.info(f"Resuming from {g_checkpoint} and {d_checkpoint}")

            G_checkpoint = torch.load(
                g_checkpoint, map_location=accelerator.device, weights_only=True
            )
            D_checkpoint = torch.load(
                d_checkpoint, map_location=accelerator.device, weights_only=True
            )

            if "epoch" in G_checkpoint:
                finished_epoch = int(G_checkpoint["epoch"])
            try:
                finished_epoch = int(Path(g_checkpoint).stem.split("_")[1])
            except:
                finished_epoch = 0

            scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_G, gamma=lr_decay, last_epoch=finished_epoch - 1
            )
            scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_D, gamma=lr_decay, last_epoch=finished_epoch - 1
            )

            G.load_state_dict(G_checkpoint["model"])
            if "optimizer" in G_checkpoint:
                optimizer_G.load_state_dict(G_checkpoint["optimizer"])
            if "scheduler" in G_checkpoint:
                scheduler_G.load_state_dict(G_checkpoint["scheduler"])

            D.load_state_dict(D_checkpoint["model"])
            if "optimizer" in D_checkpoint:
                optimizer_D.load_state_dict(D_checkpoint["optimizer"])
            if "scheduler" in D_checkpoint:
                scheduler_D.load_state_dict(D_checkpoint["scheduler"])
        else:
            finished_epoch = 0
            scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_G, gamma=lr_decay, last_epoch=-1
            )
            scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_D, gamma=lr_decay, last_epoch=-1
            )

        G, D, optimizer_G, optimizer_D = accelerator.prepare(
            G, D, optimizer_G, optimizer_D
        )

        return G, D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, finished_epoch

    def setup_dataloader(
        self,
        dataset: Dataset,
        batch_size=1,
        shuffle=True,
        accelerator: Accelerator | None = None,
    ):
        if accelerator is None:
            accelerator = Accelerator()

        dataset = dataset.with_format("torch", device=accelerator.device)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=TextAudioCollateMultiNSFsid(),
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
        )
        loader = accelerator.prepare(loader)
        return loader

    def run(
        self,
        G,
        D,
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        finished_epoch,
        loader_train,
        loader_test,
        accelerator: Accelerator | None = None,
        epochs=100,
        segment_size=17280,
        filter_length=N_FFT,
        hop_length=HOP_LENGTH,
        n_mel_channels=N_MELS,
        win_length=WIN_LENGTH,
        mel_fmin=0.0,
        mel_fmax: float | None = None,
        c_mel=45,
        c_kl=1.0,
        upload_to_hub: str | None = None,
    ):
        if accelerator is None:
            accelerator = Accelerator()

        if accelerator.is_main_process:
            logger.info("Start training")

        prev_loss_gen = -1.0
        prev_loss_fm = -1.0
        prev_loss_mel = -1.0
        prev_loss_kl = -1.0
        prev_loss_disc = -1.0
        prev_loss_gen_all = -1.0

        with accelerator.autocast():
            epoch_iterator = tqdm(
                range(1, epochs + 1),
                desc="Training",
                disable=not accelerator.is_main_process,
            )
            for epoch in epoch_iterator:
                if epoch <= finished_epoch:
                    continue

                G.train()
                D.train()

                epoch_loss_gen = 0.0
                epoch_loss_fm = 0.0
                epoch_loss_mel = 0.0
                epoch_loss_kl = 0.0
                epoch_loss_disc = 0.0
                epoch_loss_gen_all = 0.0
                num_batches = 0

                batch_iterator = tqdm(
                    loader_train,
                    desc=f"Epoch {epoch}",
                    leave=False,
                    disable=not accelerator.is_main_process,
                )
                for batch in batch_iterator:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = batch

                    # Generator
                    optimizer_G.zero_grad()
                    (
                        y_hat,
                        ids_slice,
                        x_mask,
                        z_mask,
                        (z, z_p, m_p, logs_p, m_q, logs_q),
                    ) = G(
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        sid,
                    )
                    mel = spec_to_mel_torch(
                        spec,
                        filter_length,
                        n_mel_channels,
                        self.sr,
                        mel_fmin,
                        mel_fmax,
                    )
                    y_mel = commons.slice_segments(
                        mel, ids_slice, segment_size // hop_length
                    )
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1),
                        filter_length,
                        n_mel_channels,
                        self.sr,
                        hop_length,
                        win_length,
                        mel_fmin,
                        mel_fmax,
                    )
                    wave = commons.slice_segments(
                        wave, ids_slice * hop_length, segment_size
                    )

                    # Discriminator
                    optimizer_D.zero_grad()
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = D(wave, y_hat.detach())

                    # Update Discriminator
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    accelerator.backward(loss_disc)
                    optimizer_D.step()

                    # Re-compute discriminator output (since we just got a "better" discriminator)
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = D(wave, y_hat)

                    # Update Generator
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                    accelerator.backward(loss_gen_all)
                    optimizer_G.step()

                    prev_loss_gen = loss_gen.item()
                    prev_loss_fm = loss_fm.item()
                    prev_loss_mel = loss_mel.item()
                    prev_loss_kl = loss_kl.item()
                    prev_loss_disc = loss_disc.item()
                    prev_loss_gen_all = loss_gen_all.item()

                    # Update progress bar with current losses
                    if accelerator.is_main_process:
                        batch_iterator.set_postfix(
                            {
                                "g_loss": f"{prev_loss_gen:.4f}",
                                "d_loss": f"{prev_loss_disc:.4f}",
                                "mel_loss": f"{prev_loss_mel:.4f}",
                                "total": f"{prev_loss_gen_all:.4f}",
                            }
                        )

                    epoch_loss_gen += prev_loss_gen
                    epoch_loss_fm += prev_loss_fm
                    epoch_loss_mel += prev_loss_mel
                    epoch_loss_kl += prev_loss_kl
                    epoch_loss_disc += prev_loss_disc
                    epoch_loss_gen_all += prev_loss_gen_all
                    num_batches += 1

                scheduler_G.step()
                scheduler_D.step()

                if accelerator.is_main_process and num_batches > 0:
                    avg_gen = epoch_loss_gen / num_batches
                    avg_disc = epoch_loss_disc / num_batches
                    avg_fm = epoch_loss_fm / num_batches
                    avg_mel = epoch_loss_mel / num_batches
                    avg_kl = epoch_loss_kl / num_batches
                    avg_total = epoch_loss_gen_all / num_batches

                    logger.info(
                        f"Epoch {epoch} | "
                        f"Generator Loss: {avg_gen:.4f} | "
                        f"Discriminator Loss: {avg_disc:.4f} | "
                        f"Mel Loss: {avg_mel:.4f} | "
                        f"Total Loss: {avg_total:.4f}"
                    )

                    # Update epoch progress bar
                    epoch_iterator.set_postfix(
                        {
                            "g_loss": f"{avg_gen:.4f}",
                            "d_loss": f"{avg_disc:.4f}",
                            "total": f"{avg_total:.4f}",
                        }
                    )

                    self.writer.add_scalar("Loss/Generator", avg_gen, epoch)
                    self.writer.add_scalar("Loss/Feature_Matching", avg_fm, epoch)
                    self.writer.add_scalar("Loss/Mel", avg_mel, epoch)
                    self.writer.add_scalar("Loss/KL", avg_kl, epoch)
                    self.writer.add_scalar("Loss/Discriminator", avg_disc, epoch)
                    self.writer.add_scalar("Loss/Generator_Total", avg_total, epoch)
                    self.writer.add_scalar(
                        "Learning_Rate/Generator",
                        scheduler_G.get_last_lr()[0],
                        epoch,
                    )
                    self.writer.add_scalar(
                        "Learning_Rate/Discriminator",
                        scheduler_D.get_last_lr()[0],
                        epoch,
                    )

                if loader_test is not None:
                    with torch.no_grad():
                        sample_idx = 0
                        test_iterator = tqdm(
                            loader_test,
                            desc=f"Testing epoch {epoch}",
                            leave=False,
                            disable=not accelerator.is_main_process,
                        )
                        for batch_idx, (
                            phone,
                            phone_lengths,
                            pitch,
                            pitchf,
                            spec,
                            spec_lengths,
                            wave,
                            wave_lengths,
                            sid,
                        ) in enumerate(test_iterator):
                            # Generate audio for each sample in the batch
                            audio_segments = G.infer(
                                phone, phone_lengths, pitch, pitchf, sid
                            )[0]

                            # Log each audio sample in the batch
                            for i, audio in enumerate(audio_segments):
                                audio_numpy = audio[0].data.cpu().float().numpy()
                                self.writer.add_audio(
                                    f"Audio/{sample_idx}",
                                    audio_numpy,
                                    epoch,
                                    sample_rate=self.sr,
                                )
                                sample_idx += 1

                res = TrainingCheckpoint(
                    epoch,
                    G,
                    D,
                    optimizer_G,
                    optimizer_D,
                    scheduler_G,
                    scheduler_D,
                    prev_loss_gen,
                    prev_loss_fm,
                    prev_loss_mel,
                    prev_loss_kl,
                    prev_loss_gen_all,
                    prev_loss_disc,
                )

                res.save(self.exp_dir)
                G.save_pretrained(self.exp_dir)

                if upload_to_hub is not None:
                    self.push_to_hub(upload_to_hub)
                    G.push_to_hub(upload_to_hub)

    def train(
        self,
        resume_from: Tuple[str, str] | None = None,
        accelerator: Accelerator | None = None,
        batch_size=1,
        epochs=100,
        lr=1e-4,
        lr_decay=0.999875,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps=1e-9,
        use_spectral_norm=False,
        segment_size=17280,
        filter_length=N_FFT,
        hop_length=HOP_LENGTH,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock: Literal["1", "2"] = "1",
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=512,
        upsample_rates: list[int] = [12, 10, 2, 2],
        upsample_kernel_sizes: list[int] = [24, 20, 4, 4],
        spk_embed_dim=109,
        gin_channels=256,
        n_mel_channels=N_MELS,
        win_length=WIN_LENGTH,
        mel_fmin=0.0,
        mel_fmax: float | None = None,
        c_mel=45,
        c_kl=1.0,
        upload_to_hub: str | None = None,
    ):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if accelerator is None:
            accelerator = Accelerator()

        (
            G,
            D,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            finished_epoch,
        ) = self.setup_models(
            resume_from=resume_from or self.latest_checkpoint(),
            accelerator=accelerator,
            lr=lr,
            lr_decay=lr_decay,
            betas=betas,
            eps=eps,
            use_spectral_norm=use_spectral_norm,
            segment_size=segment_size,
            filter_length=filter_length,
            hop_length=hop_length,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            spk_embed_dim=spk_embed_dim,
            gin_channels=gin_channels,
        )

        loader_train = self.setup_dataloader(
            self.dataset_train,
            batch_size=batch_size,
            accelerator=accelerator,
        )

        loader_test = (
            self.setup_dataloader(
                self.dataset_test,
                batch_size=batch_size,
                accelerator=accelerator,
                shuffle=False,
            )
            if self.dataset_test is not None
            else None
        )

        return self.run(
            G,
            D,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            finished_epoch,
            loader_train,
            loader_test,
            accelerator,
            epochs=epochs,
            segment_size=segment_size,
            filter_length=filter_length,
            hop_length=hop_length,
            n_mel_channels=n_mel_channels,
            win_length=win_length,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            c_mel=c_mel,
            c_kl=c_kl,
            upload_to_hub=upload_to_hub,
        )

    def push_to_hub(self, repo: str, private: bool = True):
        if not os.path.exists(self.exp_dir):
            raise FileNotFoundError("exp_dir not found")

        api = HfApi()
        repo_id = api.create_repo(repo_id=repo, private=private, exist_ok=True).repo_id

        return upload_folder(
            repo_id=repo_id,
            folder_path=self.exp_dir,
            commit_message="Upload via ZeroRVC",
        )

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()
