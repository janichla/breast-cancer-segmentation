# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image


def main(tempdir):
    # printed aktuelle Konfigurationsdetails der MONAI-Bibliothek in die Konsole
    monai.config.print_config()
    # initialisiert das Logging -> Nachrichten ab Schweregrad INFO werden im Terminal angezeigt
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # formatierte Ausgabe mit tempdir (Pfad zur temorären Datei)
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    
    # generiert synthetische Bilder und die dazugehörigen (binäre) Segmentierungsmasken als NumPy-Array
    # Intensitätsspektrum von Bild und Maske: [0,1]
    for i in range(40):
        im, seg = create_test_image_2d(128, 128, num_seg_classes=1)
        # Umwandlung in Wertespektrum [0,255] und dann Konvertierung in Datentyp uint8
        # Umwandlung von Numpy-Array in Bild
        # Speicherung der Bilder und Masken als png-Datei im temporären Verzeichnis tempdir und erstellt dazu den Pfad
        Image.fromarray((im * 255).astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
        Image.fromarray((seg * 255).astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))

    # lädt die Pfade der gespeicherten Bilder/Masken
    images = sorted(glob(os.path.join(tempdir, "img*.png")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.png")))

    # Preprocessing-Pipeline: 
    train_imtrans = Compose(
        [
            # lädt Bild in NumPy-Array
            LoadImage(image_only=True, ensure_channel_first=True),
            # skaliert die Pixelwerte auf den Standardbereich [0, 1]
            ScaleIntensity(),
            # schneidet zufällige Regionen (Crops) aus
            RandSpatialCrop((96, 96), random_size=False),
            # zufällige 90-Grad-Rotation 
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    train_segtrans = Compose(
        [
            # lädt Bild in NumPy-Array
            LoadImage(image_only=True, ensure_channel_first=True),
            # skaliert die Pixelwerte auf den Standardbereich [0, 1]
            ScaleIntensity(),
            # schneidet zufällige Regionen (Crops) aus
            RandSpatialCrop((96, 96), random_size=False),
            # zufällige 90-Grad-Rotation
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    # definiert Transformation: lädt Bild als NumPy-Array; skaliert Pixelwerte in [0,1]
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    # erstellt Datenset: Bild, Preprocessing des Bildes, Segmentierungsmarke, Preprocessing der Maske
    check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
    # erstellt DataLoader: lädt Daten aus check_ds in Batches
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    # extrahiert ersten Batch
    im, seg = monai.utils.misc.first(check_loader)
    # Ausgabe der Formen zur Überprüfung
    print(im.shape, seg.shape)

    # erstellt Trainingsdataset aus den ersten 20 Bildern und Masken
    train_ds = ArrayDataset(images[:20], train_imtrans, segs[:20], train_segtrans)
    # erstellt Trainings-DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    # erstellt Validationdataset aus den letzten 20 Bildern und Masken
    val_ds = ArrayDataset(images[-20:], val_imtrans, segs[-20:], val_segtrans)
    # erstellt Validierungs-DataLoader
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    # berechnet Dice-Koeffizient
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # Postprocessing:
    # Sigmoid-Aktivierungsfunktion: transformiert die vorhergesagten Werte in [0, 1]
    # Threshold: Pixelwerte über 0.5 -> 1 (Segmentierung); darunter -> 0 (Hintergrund)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # definiert Gerät (GPU oder CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modell
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    # verschiebt das Modell auf GPU/CPU
    ).to(device)
    # definiert Verlustfunktion
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    # definiert Optimierungsalgorithmus und Lernrate
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # Training
    # Validierung alle 2 Durchläufe
    val_interval = 2
    # Beste Metrik
    best_metric = -1
    # Durchlauf der besten Metrik
    best_metric_epoch = -1
    # Liste zur Speicherung des Verlustes pro Durchlauf
    epoch_loss_values = list()
    # Liste zur Speicherung der Validierungsmetriken
    metric_values = list()
    # Protokollieren der Trainingsverluste
    writer = SummaryWriter()
    # 10 Durchläufe
    for epoch in range(10):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        # versetzt Modell in Trainingsmodus
        model.train()
        # Gesamtverlust eines Durchlaufs
        epoch_loss = 0
        # Anzahl der Schritte (Batches) pro Durchlauf
        step = 0
        # 
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

# Hauptfunktion
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)