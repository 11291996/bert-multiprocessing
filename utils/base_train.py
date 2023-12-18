import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import time

def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

def multigpu(epoch, batch_size):

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
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    seed_val = 42

    model = BertForSequenceClassification.from_pretrained(
                        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                        num_labels = 2, # The number of output labels--2 for binary classification.
                                        # You can increase this for multi-class tasks.   
                        output_attentions = False, # Whether the model returns attentions weights.
                        output_hidden_states = False, # Whether the model returns all hidden-states.
                    )
    
    train_dataloader = DataLoader(
                        train_dataset,  # The training samples.
                        sampler = RandomSampler(train_dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )
    
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

    total_steps = len(train_dataloader) * epoch

    optimizer = AdamW(model.parameters(),
            lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0, # Default value in run_glue.py
                                        num_training_steps = total_steps
    )

    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, validation_dataloader, scheduler = \
    accelerator.prepare(model, optimizer, train_dataloader, validation_dataloader, scheduler)
    start_time = time.time()

    for epoch_i in range(epoch):
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc = "train_step", mininterval=0.01, disable=not accelerator.is_local_main_process)):

            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            model.zero_grad()

            output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
            loss = output.loss
            logits = output.logits

            accelerator.backward(loss)
            loss = accelerator.gather(loss).sum()
            total_train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        accelerator.print("")
        accelerator.print("  Average training loss: {0:.2f}".format(avg_train_loss))

        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader, desc = "eval_step", mininterval=0.01, disable=not accelerator.is_local_main_process):

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
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
                loss = accelerator.gather(loss).sum()

            # Accumulate the validation loss. 
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        accelerator.print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)


        accelerator.print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    end_time = time.time()

    accelerator.print("")
    accelerator.print("Training complete!")
    accelerator.print("Time taken: {}".format(end_time - start_time))