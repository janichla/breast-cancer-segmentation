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

## Bibliotheken zum Protokollieren von Nachrichten
import logging
import sys
## Arbeiten mit temporären Dateien
import os
import tempfile
## Suchen von Dateien basierend auf Mustern
from glob import glob

## Pytorch
import torch
## Speichern von Bilddaten
from PIL import Image

## Monai
from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity


def main(tempdir):
    # printed aktuelle Konfigurationsdetails der MONAI-Bibliothek in die Konsole
    config.print_config()
    # initialisiert das Logging -> Nachrichten ab Schweregrad INFO werden im Terminal angezeigt
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # formatierte Ausgabe mit tempdir (Pfad zur temorären Datei)
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    
    # generiert synthetische Bilder (im) und die dazugehörigen (binäre) Segmentierungsmasken (seg) und speichert sie als png im temorären Verzeichnis
    # Intensitätsspektrum von Bild und Maske: [0,1]
    for i in range(5):
        im, seg = create_test_image_2d(128, 128, num_seg_classes=1)
        # Umwandlung in Wertespektrum [0,255] und dann Konvertierung in Datentyp uint8
        # Umwandlung von Numpy-Array in Bild
        # Speicherung der Bilder und Masken als png-Datei im temporären Verzeichnis tempdir und erstellt dazu den Pfad
        Image.fromarray((im * 255).astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
        Image.fromarray((seg * 255).astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))

    # lädt die Pfade der gespeicherten Bilder/Masken
    images = sorted(glob(os.path.join(tempdir, "img*.png")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.png")))

    # lädt Bild; skaliert Pixelwerte in [0,1]
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    # Erstellung eines Datensets: lädt jeweils ein Bild und eine Segmentierungsmaske, wendet darauf je die Transformation an und speichert sie als Paar
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    
    # lädt das Datenset in Mini-Batches, die vom Modell verarbeitet werden
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    # berechnet Dice-Koeffizient, um Qualität der Segmentierung zu Messen
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # Nachbearbeitung der Modellvorhersagen: Sigmoid-Funktion (Transformierung in [0,1]), weil UNet eine Regressionsausgabe liefert
    # Werte unter 0,5 gelten als Hintergrund
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # Speicherung der segmentierten Daten als png 
    saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    # Geräteauswahl (GPU oder CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modell: UNet, 2 Dimensionen, Eingangs- und Ausgangsdatentyp haben 1 Kanal, Anzahl der Filter der Encoderstufen, Downsampling-Schritte des Encoders, Anzahl der Residualblöcke pro Stufe -> wird auf Gerät verschoben
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Laden des trainierten Modells
    model.load_state_dict(torch.load("best_metric_model_segmentation2d_array.pth"))
    # Versetzt das Modell in den Evaluierungsmodus
    model.eval()
    # Deaktivierung der Berechnung von Gradienten
    with torch.no_grad():
        # Iteration über alle Mini-Batches im val_loader
        for val_data in val_loader:
            # verschiebt Eingabebilder und Segmentierungslabels auf Gerät
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # Sliding Window: Bei großen Eingabebildern kann die GPU nicht genug Speicher haben, um das gesamte Bild auf einmal zu verarbeiten
            # daher Unterteilung in kleinere ROI
            roi_size = (96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            # Aufteilung von val_outputs und val_labels in einzelne Bilder zur separaten Verarbeitung
            # Anwendung der Post-Processing-Schritte zur Erzeugung binärer Segmentierung
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # Berechnet Dice-Koeffizient zwischen den vorhergesagten (val_outputs) und tatsächlichen (val_labels) Werten
            dice_metric(y_pred=val_outputs, y=val_labels)
            for val_output in val_outputs:
                saver(val_output)
        # Berechnet den endgültigen durchschnittlichen Dice-Koeffizienten über alle Bilder und gibt einen numerischen Wert zurück
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()

## Hauptfunktion
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)