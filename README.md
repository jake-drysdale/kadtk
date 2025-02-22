# Kernel Audio Distance Toolkit
The Kernel Audio Distance Toolkit (KADTK) provides an efficient and standardized implementation of Kernel Audio Distance (KAD)—a distribution-free, unbiased, and computationally efficient metric for evaluating generative audio.

## 1. Installation

To use the KAD toolkit, you must first install it. This library is created and tested on Python 3.10 on Linux but should work on Python >3.9.

### 1.1 Install
Install torch https://pytorch.org/. Only pytorch~=2.1.x officially supported.
Ensure your device is GPU-compatible and install the necessary software for CUDA support.

Ensure you have Poetry installed (Poetry Docs). Then, run:
```sh
poetry install
```

If you prefer a manual installation, first install PyTorch (pytorch.org), then run:
```sh
pip install -r requirements.txt
```

### 1.2 Troubleshooting
if scipy causes some error, reinstall scipy: pip uninstall scipy && pip install scipy==1.11.2
if charset causes some error, (re)install chardet: pip install chardet
Optional Dependencies
Optionally, you can install dependencies that add additional embedding support.



## 2. Usage
The toolkit provides a CLI command for computing KAD scores. It automatically extracts target embeddings and computes the KAD score between your generated audio files and an evaluation dataset.
```sh
kadtk <model_name> <target-set dir> <evaluation-set dir> [--fad/--inf/--indiv]
```
*--fad* TODO(description. this option is for FAD calculation. if you turn on this, the code will calculate FAD, not KAD) <br/>
*--inf* option uses metric-inf extrapolation, and *--indiv* calculates metric for individual audios. <br/>
*--force_emb_calc* forces re-calculation of embeddings. <br/>
*--audio_len* (sec) checks if the audio match the given length.


### Supported Models

| Model | Name in KADtk | Description | Creator |
| --- | --- | --- | --- |
| [CLAP](https://github.com/microsoft/CLAP) | `clap-2023` | Learning audio concepts from natural language supervision | Microsoft |
| [CLAP](https://github.com/LAION-AI/CLAP) | `clap-laion-{audio/music}` | Contrastive Language-Audio Pretraining | LAION |
| [Encodec](https://github.com/facebookresearch/encodec) | `encodec-emb` | State-of-the-art deep learning based audio codec | Facebook/Meta Research |
| [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) | `MERT-v1-95M-{layer}` | Acoustic Music Understanding Model with Large-Scale Self-supervised Training | m-a-p |
| [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md) | `vggish` | Audio feature classification embedding | Google |
| [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn/README.md) | `panns-cnn14-{16k/32k}, panns-wavegram-logmel` | PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition | Kong, Qiuqiang, et al. |
| [OpenL3](https://github.com/marl/openl3/README.md) | `openl3-{mel256/mel128}-{env/music}` | Look, Listen and Learn More: Design Choices for Deep Audio Embeddings | Cramer, Aurora et al. |
| [PaSST](https://github.com/kkoutini/passt_hear21/README.md) | `passt-{base-{10s/20s/30s}, passt-openmic, passt-fsd50k` (10s default, base for AudioSet) | Efficient Training of Audio Transformers with Patchout | Koutini, Khaled et al. |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | `dac-44kHz` | High-Fidelity Audio Compression with Improved RVQGAN | Descript |
| [CDPAM](https://github.com/pranaymanocha/PerceptualAudio) | `cdpam-{acoustic/content}` | Contrastive learning-based Deep Perceptual Audio Metric | Pranay Manocha et al. |
| [Wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) | `w2v2-{base/large}` | Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Facebook/Meta Research |
| [HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) | `hubert-{base/large}` | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | Facebook/Meta Research |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | `wavlm-{base/base-plus/large}` | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | Microsoft |
| [Whisper](https://github.com/openai/whisper) | `whisper-{tiny/base/small/medium/large}` | Robust Speech Recognition via Large-Scale Weak Supervision | OpenAI |


### Optional Dependencies

Optionally, you can install dependencies that add additional embedding support. They are:

* PANNS(updated): `pip install git+https://github.com/DCASE2024-Task7-Sound-Scene-Synthesis/fadtk.git`
* CDPAM: `pip install cdpam`
* DAC: `pip install descript-audio-codec==1.0.0`



#### Example 1: Computing KAD scores on Dev Set
1. Download Dev Set and unzip
```sh
mkdir path/to/dev
cd path/to/dev
wget https://zenodo.org/records/10869644/files/captions.csv https://zenodo.org/records/10869644/files/embeddings.tar.xz
tar -xf embeddings.tar.xz
```
2. Before computing KAD, make sure the Dev Set directory looks like this:
```
path/to/dev/
├── embeddings/
│   ├── panns-wavegram-logmel
│   ├── vggish
│   ├── clap-2023
│   └── ...
└── caption.csv # not used for calculation, just for your interest.
path/to/eval/
├── embeddings/
│   └── ...
├── stats/
│   └── ...
└── caption.csv
```
3. Calculate KAD score
```sh
# Compute KAD between the baseline and evaluation datasets on two different models
kadtk panns-wavegram-logmel /path/to/dev /path/to/evaluation/audio
kadtk vggish /path/to/dev /path/to/evaluation/audio --fad
```

#### Example 2: Compute individual KAD scores for each audio

```sh
kadtk encodec-emb /path/to/baseline/audio /path/to/evaluation/audio --indiv scores.csv
```

#### Example 3: Compute KAD scores with your own baseline

First, create two directories, one for the baseline and one for the evaluation, and place *only* the audio files in them. Then, run the following commands:

```sh
# Compute KAD between the baseline and evaluation datasets
kadtk clap-laion-audio /path/to/baseline/audio /path/to/evaluation/audio
```

#### Example 4: Just compute embeddings

If you only want to compute embeddings with a list of specific models for a list of dataset, you can do that using the command line.

```sh
kadtk.embeds -m Model1 Model2 -d /dataset1 /dataset2
```

## 0x06. Citation, Acknowledgments and Licenses

```latex
@article{fad_embeddings,
    author = {Tailleur, Modan and Lee, Junwon and Lagrange, Mathieu and Choi, Keunwoo and Heller, Laurie M. and Imoto, Keisuke and Okamoto, Yuki},
    title = {Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant},
    journal = {arXiv:2403.17508},
    url = {https://arxiv.org/abs/2403.17508},
    year = {2024}
}
```

```latex
@inproceedings{fadtk,
  title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
  author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
  booktitle = {Proc. IEEE ICASSP 2024},
  year = {2024},
  url = {https://arxiv.org/abs/2311.01616},
}
```