import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from modules.parallel_bert import PBertForSequenceClassification
import numpy as np
from tqdm import tqdm
import time

class trainer:
    def __init__(self) -> None:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        df = pd.read_csv("/scratch/paneah/cola_public/raw/in_domain_train.tsv", \
                 delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

        #get the lists of sentences and their labels.
        sentences = df.sentence.values
        labels = df.label.values

        #tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 64,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )

            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        #convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        #combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        #calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        #divide the dataset by randomly selecting samples.
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
        
    def training(self, model, epoch, batch_size, device_name):
        #create the DataLoaders for our training and validation sets.
        #we'll take training samples in random order. 
        model = self.model
        self.train_dataloader = DataLoader(
                        self.train_dataset,  # The training samples.
                        sampler = RandomSampler(self.train_dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )
        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.validation_dataloader = DataLoader(
                    self.val_dataset, # The validation samples.
                    sampler = SequentialSampler(self.val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

        self.seed_val = 42

        self.total_steps = len(self.train_dataloader) * epoch

        self.optimizer = AdamW(model.parameters(),
              lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
              eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = self.total_steps
        )
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)
        if device_name == "cuda":
            device = torch.device("cuda")
            self.model.to(device)
        start_time = time.time()
        for epoch_i in range(epoch):
            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc = "train_step", mininterval=0.01)):
                if device_name == "cuda":
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                else: 
                    b_input_ids = batch[0]
                    b_input_mask = batch[1]
                    b_labels = batch[2]

                self.model.zero_grad()

                output = self.model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask, 
                                     labels=b_labels)
                loss = output.loss
                logits = output.logits

                loss.backward()
                    
                total_train_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                self.optimizer.step()

                self.scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            self.model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in tqdm(self.validation_dataloader, desc = "eval_step", mininterval=0.01):

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                if device_name == "cuda":
                    device = torch.device("cuda")
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                else: 
                    b_input_ids = batch[0]
                    b_input_mask = batch[1]
                    b_labels = batch[2]

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        
                
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                    loss = output.loss
                    logits = output.logits
                
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)


            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)


            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        end_time = time.time()

        print("")
        print("Training complete!")
        print("Time taken: {}".format(end_time - start_time))


    def train(self, model_name, epoch, batch_size, device):
        if model_name == "bert":
            self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                    num_labels = 2, # The number of output labels--2 for binary classification.
                                    # You can increase this for multi-class tasks.   
                    output_attentions = False, # Whether the model returns attentions weights.
                    output_hidden_states = False, # Whether the model returns all hidden-states.
                )
            if device == "cuda":
                self.training(self.model, epoch, batch_size, device_name = "cuda")
            else: 
                self.training(self.model, epoch, batch_size, device_name = "None")

        elif model_name == "pbert":
                self.model = PBertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                    num_labels = 2, # The number of output labels--2 for binary classification.
                                    # You can increase this for multi-class tasks.   
                    output_attentions = False, # Whether the model returns attentions weights.
                    output_hidden_states = False, # Whether the model returns all hidden-states.
                )
        
                self.model.update_weight()

                self.training(self.model, epoch, batch_size, device_name = None)