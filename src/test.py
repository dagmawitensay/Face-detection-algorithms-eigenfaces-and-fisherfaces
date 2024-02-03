import numpy as np
import time
from mltu.tokenizers import CustomTokenizer
from mltu.inferenceModel import OnnxInferenceModel

class PtEnTranslator(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.new_inputs = self.model.get_inputs()
        # print("amharic file", self.metadata["tokenizer"])
        print("geez file", self.metadata["tokenizer"]["char_level"])
        self.tokenizer = CustomTokenizer.load(self.metadata["tokenizer"])
        self.detokenizer = CustomTokenizer.load(self.metadata["detokenizer"])
        # self.tokenizer.char_level = False
        # self.detokenizer.char_level = False

    def predict(self, sentence):
        start = time.time()
        tokenized_sentence = self.tokenizer.texts_to_sequences([sentence])[0]
        # tokenized_sentence = [x for x in tokenized_sentence if x != 3]
        print("tokenized sentence", self.tokenizer.char_level, tokenized_sentence)
        encoder_input = np.pad(tokenized_sentence, (0, self.detokenizer.max_length - len(tokenized_sentence)), constant_values=0).astype(np.int64)
        
        tokenized_results = [self.detokenizer.start_token_index]

        for index in range(self.detokenizer.max_length - 1):
            decoder_input = np.pad(tokenized_results, (0, self.tokenizer.max_length - len(tokenized_results)), constant_values=0).astype(np.int64)
            input_dict = {
                self.model._inputs_meta[0].name: np.expand_dims(encoder_input, axis=0),
                self.model._inputs_meta[1].name: np.expand_dims(decoder_input, axis=0),
            }
            # print("input_dict", input_dict["input_1"].shape, decoder_input.shape)
            preds = self.model.run(None, input_dict)[0]  # preds shape (1, seq_length, vocab_size)
            pred_results = np.argmax(preds, axis=2)
            tokenized_results.append(pred_results[0][index])

            if tokenized_results[-1] == self.detokenizer.end_token_index:
                break
        print("tokenezed results", tokenized_results)
        results = self.detokenizer.detokenize([tokenized_results])
        return results[0], time.time() - start