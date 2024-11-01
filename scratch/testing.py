from mfccProcessor import MFCCProcessor
from preprocessing import GoogleSpeechDataset

processor = MFCCProcessor()
dataset = GoogleSpeechDataset('google_speech/visual', processor)

# Print dataset length
print(f'Total samples: {len(dataset)}')

# Access the first sample
feature, label = dataset[0]
print(f'Feature shape: {feature.shape}, Label: {label}')

'''
Returns:
Total samples: 1592
Feature shape: torch.Size([13, 44]), Label: 0

Process finished with exit code 0
'''