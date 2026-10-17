# ADHD Classification Using Convolutional Neural Networks Based on EEG Spectrograms

## Project overview

The objective of this project was to evaluate whether convolutional neural networks (CNNs) can distinguish adults with ADHD from healthy controls using time-frequency representations of EEG signals.

The project uses Short-Time Fourier Transform (STFT) spectrograms as image inputs to deep learning models.

The experiments investigate two main factors:

- **EEG channel configuration:** spectrograms generated from 19 channels or a selected 4 channels.
- **Recording condition:** EEG signals recorded with eyes open (EO) or eyes closed (EC).

Five CNN architectures are evaluated to compare their classification performance:

* **custom 4-layer CNN**
* **custom 10-layer CNN**
* **ResNet50V2**
* **InceptionV3**
* **VGG16**

## Dataset

The experiments use EEG recordings from the [TDBrain database](https://brainclinics.com/resources/tdbrain-dataset), which contains EEG data from individuals with different neuropsychiatric conditions and healthy controls.

The project uses a subset consisting of healthy adults and patients with ADHD older than 21.

| Property | Description |
| --- | --- |
| Dataset | TDBrain |
| Total participants | 77 |
| ADHD group | 40 participants |
| Control group | 37 participants |
| Recording conditions | Eyes Open (EO), Eyes Closed (EC) |
| Original EEG channels | 26 |
| Channels used in the experiments | 19 or 4 |

> **Dataset access:** Please consult the dataset provider's terms and conditions. This repository does not imply ownership of, or unrestricted redistribution rights to, the original EEG recordings.

## Methodology

### EEG Preprocessing

The EEG recordings were preprocessed before being used in the experiments. The preprocessing pipeline included:

* Raw EEG data acquisition,
* Independent Component Analysis (ICA) for artifact removal,
* Laplacian reference for spatial filtering,
* Band-pass filtering between 0.5 Hz and 40 Hz,
* Notch filtering at 50 Hz to suppress power-line interference,
* Segmentation into 10-second intervals,
* Spectrogram generation.

<p align="center">
  <img src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/data_preprocessing/preprocessing_pipeline.png"/>
</p>

### Spectrogram Generation

Each EEG segment is transformed into a time-frequency representation using the Short-Time Fourier Transform.

The spectrogram generation procedure uses:

* a 2-second Hamming window,
* a 90% overlap between consecutive windows,
* averaging of spectral power across the selected EEG channel,
* logarithmic scaling of spectral power,
* image resizing to 224 × 224 pixels.

Two channel configurations are investigated:

**19-channel configuration** on the left and **4-channel configuration** on the right
Both spectrograms correspond to the same subject and the same time segment.

<p align="center">
  <img width="50%" alt="19-channel spectrogram example" src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/data_preprocessing/19ch_example_spectrogram.png"/> <img width="50%" alt="4-channel spectrogram example" src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/data_preprocessing/4ch_example_spectrogram.png"/>
</p>

In the first session, the models were trained on spectrograms generated from 19 channels of the standard 10–20 electrode system. In the second session, the models were trained on spectrograms generated using only four selected channels: F3, Fz, F4, and Cz. The selection of these channels was motivated by the characteristics of the theta-to-beta ratio (TBR), which is most prominent over the frontal and central cortical regions.

Each configuration is evaluated separately for EO and EC recordings. Ultimately, the dataset consisted of 924 spectrograms for each recording condition and channel configuration.

### CNN Architectures

Five convolutional neural network architectures are compared.

| Model | Description |
| --- | --- |
| CNN-4 | Custom CNN with four convolutional layers |
| CNN-10 | Custom CNN with ten convolutional layers |
| ResNet50V2 | Residual CNN pretrained on ImageNet |
| VGG16 | Deep CNN pretrained on ImageNet |
| InceptionV3 | CNN using Inception modules, pretrained on ImageNet |

### Training Configuration

The models are trained using TensorFlow and Keras.

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Batch size | 16 |
| Maximum epochs | 50 |
| Cross-validation | 5-fold |
| Random seed | Fixed for reproducibility |

Binary cross-entropy is used for CNN-4, CNN-10 and VGG16. Binary focal cross-entropy is used for ResNet50V2 and InceptionV3.

### Patient-Level Evaluation

All spectrograms originating from the same participant are assigned to the same fold. This prevents recordings from a single participant from appearing in both the training and test sets.

For each participant, the model produces a probability for every spectrogram. These probabilities are averaged to obtain a single patient-level prediction. A mean probability greater than 0.5 corresponds to the ADHD class; otherwise, the participant is assigned to the control group.

## Evaluation Metrics

Model performance is assessed using the following classification metrics:

* **Accuracy:** proportion of correctly classified participants,
* **Recall:** proportion of participants with ADHD correctly identified,
* **Specificity:** proportion of control participants correctly identified,
* **Precision:** proportion of positive predictions that are correct,
* **F1-score:** harmonic mean of precision and recall.

### Statistical Analysis

The Friedman test is used to assess differences between the five architectures, using patient-level accuracy values for each configuration.

If the Friedman test indicates a statistically significant difference, post-hoc multiple comparisons with Bonferroni correction are performed.

# Results

The best overall reported result was obtained by VGG16 using spectrograms generated from four EEG channels under the eyes-closed condition.

19-channel spectrogram condition:
<p align="center">
  <img width="90%" alt="19-channel results"  src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/results/EEG_19ch_results.png"/>
</p>

4-channel spectrogram condition:
<p align="center">
  <img width="90%" alt="4-channel results"  src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/results/EEG_4ch_results.png"/>
</p>

Statistical tests showed that VGG16 performed significantly differently from InceptionV3 in the four-channel EC setup (Friedman test p = 0.0269; direct comparison p = 0.0179).

<p align="center">
  <img src="https://github.com/3nrlque/EEG-ADHD-CNN-Classification/blob/main/results/Post_hoc_results.png"/>
</p>


For the other three configurations, the Friedman test did not identify statistically significant differences between the evaluated architectures at the 0.05 significance level.

The results showed that reducing the number of EEG channels used improved classification performance. The average accuracy of the models was higher for spectrograms generated from 4 channels than for those generated using all available channels. For both channel configurations, the models achieved higher average accuracy under the EC condition than under the EO condition.

The improved performance observed with spectrograms generated from 4 EEG channels may be attributed to the selection of channels containing more discriminative information for distinguishing individuals with ADHD from healthy controls. Using a smaller number of channels may also reduce the influence of irrelevant information and redundant features, allowing CNNs to focus on more task-relevant time-frequency patterns.

The higher average accuracy obtained under the EC condition may be related to differences in brain activity between eyes-closed and eyes-open states. Closing the eyes typically increases posterior alpha activity, whereas opening the eyes suppresses this rhythm and changes the spectral distribution of EEG activity. These differences may produce more distinguishable time-frequency patterns in EC spectrograms, potentially facilitating classification.

The obtained results are insufficient for the tested models to be used as standalone diagnostic tools. However, the best-performing networks, which demonstrated high sensitivity in correctly identifying individuals with ADHD, could potentially serve as objective decision-support tools to assist healthcare professionals.

## Possible improvements

* **Explore additional CNN architectures**
* **Investigate alternative EEG representations**
* **Use larger datasets**