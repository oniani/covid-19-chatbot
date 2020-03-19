# COVID-19 Chatbot

Example usage:

```sh
PYTHONPATH=src python3 train.py --dataset=data \
                                --model=345M \
                                --optimizer=adam \
                                --learning_rate=0.000005 \
                                --num_iterations=1000
```

Google Colab training example:

```python
# Mount the drive
from google.colab import drive
drive.mount("/content/drive")

# Install dependencies
!python3 -m pip install fire==0.2.1 \
                        tensorflow-gpu==1.14 \
                        tensorflow-hub==0.7.0 \
                        toposort==1.5

# Install the pretrained model
!git clone https://github.com/openai/gpt-2 "/content/drive/My Drive/gpt-2"

%cd "/content/drive/My Drive/COVID-19 CHATBOT/"
!rm -rf ./checkpoint ./models
!python3 "/content/drive/My Drive/gpt-2/download_model.py" 345M

# Run the transfer learning training
!PYTHONPATH=src python3 train.py --dataset=data \
                                 --model=345M \
                                 --optimizer=adam \
                                 --learning_rate=0.000005 \
                                 --num_iterations=10000
```
