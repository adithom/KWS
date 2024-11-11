## FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network

Adapted from https://github.com/microsoft/EdgeML/

Paper: https://arxiv.org/pdf/1901.02358

Changes made: MFCC Processor, KeywordSpotter class in train_classifier.py, CTC Loss (adapted from Pytorch source code), data pipeline, removed redundancies

### Instructions

Download the google speech dataset

```
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir google_speech
tar -xvzf speech_commands_v0.02.tar.gz -C google_speech
rm speech_commands_v0.02.tar.gz
```

Install requirements.txt (and ensure dependency compatibility(#todo))
```
pip install -r requirements.txt
```

Install setup.py
```
cd cuda
python setup.py install
```

Run Model
```
python train_classifier.py --dataset ./google_speech --epochs 30 --batch_size 64 --outdir OUTPUT_DIRECTORY --use_gpu
```