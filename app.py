# model_service.py
from flask import Flask, jsonify, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import logging
from multiprocessing import Process, Queue, Value, Lock
import os
import ctypes
import numpy as np
import psutil
# from waitress import serve
import json
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from safetensors import safe_open

# Assuming you have your SimpleNN model
def save_model_to_safetensors(model, path):
    # Convert model state dict to CPU if it's on GPU
    state_dict = model.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    
    # Save using safetensors
    save_file(state_dict, path)

# To load the model later
def load_model_from_safetensors(model_class, path, input_dim=1536, output_dim=3):
    # Create a new instance of your model
    model = model_class(input_dim, output_dim)
    
    # Load the state dict
    state_dict = load_file(path)
    
    # Load weights into model
    model.load_state_dict(state_dict)
    return model


# # Define a simple feedforward neural network
# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)  # First layer with 128 neurons
#         self.fc2 = nn.Linear(256, 1024)
#         self.fc3 = nn.Linear(1024, 256)
#         self.fc4 = nn.Linear(256, output_dim)  # Output layer
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  # Activation function for hidden layer
#         x = torch.relu(self.fc2(x))  
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)               # Output layer (logits)
#         return x
# # Add to safe globals before loading
# torch.serialization.add_safe_globals([SimpleNN])

# # Then load the model
# model = load_model_from_safetensors(SimpleNN, './multi_small/lite_cls.safetensors')
# model.eval()
    
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights and config to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config = {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **kwargs  # Additional config parameters
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save weights
        state_dict = self.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        weights_path = os.path.join(save_directory, "model.safetensors")
        save_file(state_dict, weights_path)
        
        logger.info(f"Model saved to {save_directory}")
        return config
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        # device: Optional[str] = None,
        **kwargs
    ) -> "SimpleNN":
        """
        Load pretrained model from directory
        
        Args:
            pretrained_model_path: Directory containing model.safetensors and config.json
            device: Device to load model to ('cuda', 'cpu', etc)
            **kwargs: Override config parameters
            
        Returns:
            Loaded model instance
        """
        # if device is None:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load config
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Override config with kwargs
        config.update(kwargs)
        
        # Initialize model
        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim']
        )
        
        # Load weights
        weights_path = os.path.join(pretrained_model_path, "model.safetensors")
        # state_dict = load_file(weights_path)
        # model.load_state_dict(state_dict)

        tensors = {}
        with safe_open(os.path.join(pretrained_model_path, "model.safetensors"), framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        
        # model.to(device)
        model.load_state_dict(tensors)
        model.eval()
        
        # logger.info(f"Model loaded from {pretrained_model_path} to {device}")
        return model

# Example usage
def train_and_save_model():
    # Create and train model
    model = SimpleNN(input_dim=1536, output_dim=3)
    
    # Save pretrained model
    save_dir = "./multi_small"
    model.save_pretrained(
        save_dir,
        training_params={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        version="1.0.0"
    )
    return save_dir

def load_and_use_model():
    # Load pretrained model
    model = SimpleNN.from_pretrained(
        "./multi_small",
        # device="cuda",  # optional
        # Override config parameters if needed
        input_dim=1536,
        output_dim=3
    )
    
    # Use model
    with torch.no_grad():
        dummy_input = torch.randn(1, 1536)#.to(model.device)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    return model

model = load_and_use_model()
model.eval()


class ModelWorker:
    def __init__(self, model_path, request_queue, response_queue, worker_id):
        self.model_path = model_path
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        
        # Pin to specific CPU core on Unix
        os.sched_setaffinity(0, {worker_id})
        torch.set_num_threads(1)  # Use single thread per worker
        
        self.load_model()
        
    def load_model(self):
        logger.info(f"Worker {self.worker_id}: Loading model")
        
        self.tokenizer = BertTokenizer.from_pretrained(
            'google-bert/bert-base-multilingual-cased',
            do_lower_case=False,
            strip_accents=False
        )
        
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        # JIT compile model for better performance
        # dummy_input = self.tokenizer(
        #     "dummy text",
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=64
        # )
        
        # with torch.no_grad():
        #     self.model = torch.jit.trace(
        #         self.model,
        #         # {
        #             dummy_input
        #             # dummy_input['input_ids'],
        #             # dummy_input['attention_mask'],
        #             # dummy_input['token_type_ids']
        #     # )
        #     )
        #     self.model = torch.jit.optimize_for_inference(self.model)
            
        logger.info(f"Worker {self.worker_id}: Model loaded successfully")
        
    def predict(self, sentences, request_ids):
        if isinstance(sentences, str):
            sentences = [sentences]
            request_ids = [request_ids]
            
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs['token_type_ids']
            )
            predictions = torch.softmax(outputs.logits, dim=1)
            
        # Process results
        results = []
        for i, pred in enumerate(predictions):
            predicted_class = torch.argmax(pred).item()
            confidence = pred[predicted_class].item()
            results.append({
                'request_id': request_ids[i],
                'class': predicted_class,
                'confidence': confidence,
                'worker_id': self.worker_id
            })
            
        return results

    def run(self):
        logger.info(f"Worker {self.worker_id}: Starting prediction loop")
        
        batch = []
        batch_ids = []
        last_process_time = time.time()
        
        while True:
            try:
                # Get request from queue with timeout
                try:
                    request = self.request_queue.get(timeout=0.01)
                    batch.append(request['sentence'])
                    batch_ids.append(request['request_id'])
                except:
                    pass
                
                # Process batch if it's full or timeout reached
                current_time = time.time()
                if len(batch) >= 32 or (batch and current_time - last_process_time > 0.1):
                    if batch:
                        results = self.predict(batch, batch_ids)
                        for result in results:
                            self.response_queue.put(result)
                            
                        batch = []
                        batch_ids = []
                        last_process_time = current_time
                        
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error processing batch: {str(e)}")
                for request_id in batch_ids:
                    self.response_queue.put({
                        'request_id': request_id,
                        'error': str(e),
                        'worker_id': self.worker_id
                    })
                batch = []
                batch_ids = []

def start_worker(model_path, request_queue, response_queue, worker_id):
    worker = ModelWorker(model_path, request_queue, response_queue, worker_id)
    worker.run()

# Flask application
app = Flask(__name__)



# Shared queues and worker counter
request_queue = Queue()
response_queue = Queue()
worker_counter = Value(ctypes.c_int, 0)
counter_lock = Lock()

def get_next_worker():
    """Round-robin worker selection"""
    with counter_lock:
        worker_counter.value = (worker_counter.value + 1) % 4
        return worker_counter.value

@app.route('/bert/classify/<string:sentence>', methods=['GET'])
def bert_classify(sentence):
    try:
        start_time = time.time()
        request_id = int(time.time() * 1000000)  # Microsecond timestamp as ID
        
        # Create request
        request_data = {
            'request_id': request_id,
            'sentence': sentence
        }
        
        # Send to worker queue
        request_queue.put(request_data)
        
        # Wait for result with timeout
        max_wait = 30  # seconds
        while time.time() - start_time < max_wait:
            try:
                result = response_queue.get(timeout=0.1)
                if result['request_id'] == request_id:
                    if 'error' in result:
                        return jsonify({'error': result['error']}), 500
                    
                    # Map prediction to class name
                    class_mapping = {0: 'none', 1: 'product', 2: 'series'}
                    predicted_label = class_mapping.get(result['class'], 'none')
                    
                    return jsonify({
                        'class': predicted_label,
                        'sentence': sentence,
                        'confidence': result['confidence'],
                        'processing_time': round(time.time() - start_time, 4),
                        'worker_id': result['worker_id']
                    })
                else:
                    # Put back result if it's not ours
                    response_queue.put(result)
            except:
                continue
                
        return jsonify({'error': 'Request timeout'}), 408
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500


import os, time

from openai import AzureOpenAI

client = AzureOpenAI(
    # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
    api_version= os.environ["OPENAI_API_VERSION"],
    # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME"],
)

def embed(text):
    # time.sleep(0.01)
    print(text)
    return client.embeddings.create(input = [text], model=os.environ["OPENAI_DEPLOYMENT_NAME"]).data[0].embedding




@app.route('/embed/classify/<string:sentence>', methods=['GET'])
def embed_classify(sentence):
    start_time = time.time()
    m = nn.Softmax()
    props = m(model(torch.tensor(embed(f'{sentence}'), dtype=torch.float32)))
    print(props)
    result_class = torch.argmax(props).item()
    class_mapping = {0: 'none', 1: 'product', 2: 'series'}
    predicted_label = class_mapping.get(int(result_class))
    return jsonify({
                    'class': predicted_label,
                    'sentence': sentence,
                    'confidence': torch.max(props).item(),
                    'processing_time': round(time.time() - start_time, 4),
                })


def start_server():
    # Start worker processes
    num_workers = 1
    workers = []
    
    for i in range(num_workers):
        p = Process(
            target=start_worker,
            args=('./multi_base', request_queue, response_queue, i)
        )
        p.daemon = True
        p.start()
        workers.append(p)
        
    # Start Flask server
    # serve(app, host='0.0.0.0', port=5000, threads=16, backlog=2048)
    app.run()
    
    # Clean up workers
    for p in workers:
        p.terminate()
        p.join()

if __name__ == '__main__':
    start_server()
