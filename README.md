## FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network

Adapted from https://github.com/microsoft/EdgeML/

Paper: https://arxiv.org/pdf/1901.02358

### Instructions

Clone the repository. 

```
git clone https://github.com/adithom/KWS.git
```

Install requirements.txt (and ensure dependency compatibility(#todo))
```
pip install -r requirements.txt
```

Download the google speech dataset

```
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```
```
mkdir google_speech
```
```
tar -xvzf speech_commands_v0.02.tar.gz -C google_speech
```
```
rm speech_commands_v0.02.tar.gz
```
This implementation of the FastGRNN model has trouble handling datasets as large as the Google Speech Command Dataset. So we create a subset of the dataset with 12 classes of word utterances rather than 35. The classes are: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one' and 'noise'.

```
python subdatasetCreate.py
```

Created augmented dataset from the 12 classes. Applies audio augmentation techniques to the dataset, this approach with multiple scripts is a bit redundant for dataset manipulation so this is a temporary solution.
```
python audioAugmentation.py
```

Install CUDA kernel <br>
There is an unresolved issue with the CUDA kernel currently. Working on a fix.
```
cd cuda
python setup.py install
```

#### Run Model
```
python trainClassifier.py --dataset ./google_12 --epochs 30 --batch_size 64 --outdir OUTPUT_DIRECTORY 
```

Adjust parameters in the TrainingConfig.py file to control various aspects of training. There are currently issues with inference for batch norm and training on GPU using custom CUDA acceleration.

#### Inference

Inference is currently segmented for models trained with and without batchnorm. So there are two inference scripts.