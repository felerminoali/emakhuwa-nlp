import torch

from torchtext.data.metrics import bleu_score
import sys

def translate_sentence(model, sentence, emakhuwa, portuguese, device, max_length=50):
    # Load emakhuwa tokenizer

    if type(sentence) == str:
        tokens = [token.lower() for token in sentence.replace('\n','').split()]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, emakhuwa.init_token)
    tokens.append(emakhuwa.eos_token)

    # Go through each emakhuwa token and convert to an index
    text_to_indices = [emakhuwa.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [portuguese.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder, hiddens, cells
            )
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == portuguese.vocab.stoi["<eos>"]:
            break
    
    
    translated_sentence = [portuguese.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, emakhuwa, portuguese, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["emakhuwa"]
        trg = vars(example)["portuguese"]

        prediction = translate_sentence(model, src, emakhuwa, portuguese, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    #print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])