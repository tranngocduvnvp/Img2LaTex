import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import PreTrainedTokenizerFast
import re
import json
# from torch.utils.dlpack import from_dlpack

class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])

        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, "respone_latex"
            )["data_type"]
        )
        
        self.seq_len = 256
        self.max_seq_len = 512
        self.filter_thres = 0.9
        self.temperature = 0.1
        self.eos_token = 2
        
        print("Loading model ......")
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"/workspace/tokenizer.json")
    

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            b, c, h, w = input_tensor.as_numpy().shape
            input_tensor = pb_utils.Tensor("input", input_tensor.as_numpy().astype(np.float32))

            #make request to encoder
            encoder_request = pb_utils.InferenceRequest(
                model_name="encoder",
                requested_output_names=["output"],
                inputs=[input_tensor]
            )
            encoder_respond = encoder_request.exec()
            context = pb_utils.get_output_tensor_by_name(encoder_respond, "output")
            context_tensor = pb_utils.Tensor.from_dlpack("context", context.to_dlpack())
            # get shape of context
            out = np.ones((b, 1), dtype=np.int64)
            mask = np.full_like(out, True, dtype=bool)
            b, t = out.shape
            num_dims = len(out.shape)
     
            for _ in range(self.seq_len):
                x = out[:, -self.max_seq_len:]
                mask = mask[:, -self.max_seq_len:]
                
                x_tensor = pb_utils.Tensor("x", x.astype(np.int64))
                mask_tensor = pb_utils.Tensor("mask", mask.astype(bool))

                #make request to decoder 
                decoder_request = pb_utils.InferenceRequest(
                    model_name="decoder",
                    requested_output_names=["output"],
                    inputs=[x_tensor, mask_tensor, context_tensor],
                    preferred_memory = pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU,0)
                )
                decoder_respond = decoder_request.exec()
                logits = pb_utils.get_output_tensor_by_name(decoder_respond, "output")
                logits_numpy = logits.as_numpy()
                filtered_logits = self.top_k(logits_numpy[:, -1, :], thres=self.filter_thres)
                probs = self.softmax(filtered_logits / self.temperature, axis=-1)

                sample = self.multinomial_sample(probs, 1)

                # out = torch.cat((out, sample), dim=-1)
                out = np.concatenate([out, sample], axis=-1)
                mask = self.pad_numpy(mask, (0, 1), value=True)

                if self.eos_token is not None and (np.cumsum(out == self.eos_token, 1)[:, -1] >= 1).all():
                    break
            
            out = out[:, t:]

            if num_dims == 1:
                out = out.squeeze(0)
            
            pred = self.post_process(self.token2str(out, self.tokenizer)[0])
            
            output_tensor = pb_utils.Tensor("respone_latex", np.array(pred, dtype=self.output_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
        return responses
    
    def finalize(self):
        pass
    
    def pad_numpy(self, mask, pad_width=(0, 1), value=True):
        padded_mask = np.pad(mask, ((0, 0), pad_width), mode='constant', constant_values=value)
        return padded_mask
    
    def top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        ind = np.argpartition(logits, -k, axis=-1)[:, -k:]
        
        probs = np.full_like(logits, -np.inf)
        
        row_indices = np.arange(logits.shape[0])[:, np.newaxis]
        probs[row_indices, ind] = logits[row_indices, ind]
        
        return probs
    
    def softmax(self, logits, axis=-1):
        exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

    def multinomial_sample(self, probs, num_samples=1):
        m, n = probs.shape
        cumulative_probs = np.cumsum(probs, axis=1)
        
        rand_values = np.random.rand(m, num_samples)
        rand_values = np.tile(rand_values, (1, n)).reshape(m, n, num_samples)

        samples = (rand_values < cumulative_probs[:, :, np.newaxis]).argmax(axis=1)
        
        return samples

    def post_process(self, s: str):
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
        letter = '[a-zA-Z]'
        noletter = '[\W_^\d]'
        names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
            news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
            if news == s:
                break
        return s
    
    def token2str(self, tokens, tokenizer) -> list:
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
        dec = [tokenizer.decode(tok) for tok in tokens]
        return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]
