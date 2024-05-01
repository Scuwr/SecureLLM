import torch
import torch.functional as F
from modeling.secureLLM.slora import SloraModel
from modeling.secureLLM.peft_types import PeftType_Slora
from modeling.secureLLM.slora_types import SloraLayer
from transformers import PreTrainedModel
from peft.config import PeftConfig
from peft.utils import _set_trainable
from peft.utils.other import _get_submodules

from transformers.modeling_outputs import CausalLMOutputWithPast

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType_Slora.SLORA_SUM: SloraModel,
    PeftType_Slora.SLORA_AVG: SloraModel,
    PeftType_Slora.SLORA_ABS_MAX_SUM: SloraModel,
    PeftType_Slora.SLORA_MAX_SUM: SloraModel,
    PeftType_Slora.SLORA_MAX_ELEM: SloraModel,
    PeftType_Slora.SLORA_MAX_DIFF_ELEM: SloraModel,
    PeftType_Slora.SLORA_CONCAT: SloraModel,
    PeftType_Slora.SLORA_LSE: SloraModel,
    PeftType_Slora.SLORA_SM: SloraModel,
    PeftType_Slora.SLORA_LOGIT: SloraModel,
    PeftType_Slora.SLORA_CONSENSUS: SloraModel,
    PeftType_Slora.SLORA_LORAHUB: SloraModel
}

class SloraBaseModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", adapter_weights=None):
        super().__init__()
        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.adapter_weights = adapter_weights
        if not peft_config.is_prompt_learning:
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    
    def forward(self, *args: object(), **kwargs: object()):
        """
        Forward pass of the model.
        """

        slora_size = self.peft_config[self.active_adapter].n

        if self.peft_config[self.active_adapter].peft_type == PeftType_Slora.SLORA_LOGIT:
            # print("Expanding input to match number of adapters")
            args = list(args)
            text_input = args[0]

            # Transform input from (bsz, seq_len, token_embedding) -> (adapter_idx, bsz, seq_len, token_embedding)
            text_input = torch.cat([text_input.unsqueeze(dim=0)] * slora_size, dim=0)
            args[0] = text_input
            args = tuple(args)
        
        elif self.peft_config[self.active_adapter].peft_type == PeftType_Slora.SLORA_CONSENSUS:
            args = list(args)
            text_input = args[0]
            
            text_input = torch.cat([text_input.unsqueeze(dim=0)]*(slora_size+1), dim = 0)
            args[0] = text_input
            
            args = tuple(args)

        # Output size (adapter_idx, bsz, seq_len, token_embedding)
        # For Consensus: (adapter_idx+1, bsz, seq_len, token_embedding)
        outs = self.get_base_model()(*args, **kwargs)

        if self.peft_config[self.active_adapter].peft_type == PeftType_Slora.SLORA_LOGIT:
            if type(outs) is CausalLMOutputWithPast:
                adapters = list(range(0, slora_size))
                negative_adapters = self.peft_config[self.active_adapter].negated_adapters
                positive_adapters = [x for x in adapters if x not in negative_adapters]

                #print(outs.logits.shape)
                scale = torch.ones(slora_size, 1, 1, 1).to(outs.logits.device)
                if self.adapter_weights is not None:
                    scale = scale * self.adapter_weights.reshape(scale.shape)
                outs.logits = outs.logits * scale
                
                positive_logits = torch.max(outs.logits[positive_adapters], dim=0)
                positive_logits = positive_logits.values
                # positive_logits[:, :, 1533] += 100
                

                if len(negative_adapters) > 0:
                    # method 1
                    # negative_logits = torch.max(outs.logits[negative_adapters], dim=0).values
                    # outs.logits = torch.min(positive_logits, negative_logits)
                    # method 2
                    negative_logits = torch.sum(outs.logits[negative_adapters], dim=0)
                    outs.logits = positive_logits - negative_logits
                else:
                    outs.logits = positive_logits
            else:
                raise NotImplementedError
        
        if self.peft_config[self.active_adapter].peft_type == PeftType_Slora.SLORA_CONSENSUS:
            if type(outs) is CausalLMOutputWithPast:
                adapters = list(range(0, slora_size))
                negative_adapters = self.peft_config[self.active_adapter].negated_adapters
                positive_adapters = [x+1 for x in adapters if x not in negative_adapters]

                for idx in range(0, len(negative_adapters)):
                    negative_adapters[idx] += 1

                #print(outs.logits.shape)
                scale = torch.ones(slora_size+1, 1, 1, 1).to(outs.logits.device)
                if self.adapter_weights is not None:
                    scale = scale * self.adapter_weights.reshape(scale.shape)
                outs.logits = outs.logits * scale
            
                ### Creating consensus from results. 
                '''
                Idea: Get the most likely token from each. 
                1. If all agree, modify outs so it favors this.
                2. If base and one of the adapters agree but the other dissents, this displays specialized knowledge from the other adapter and select this token.
                3. If both adapters agree and base does not, select the choice from adapters. 
                4. If none agree, defer back to the original logits method (from above.) 
                '''

                ### Remove line in final version
                return outs

                ### Finding unanimous agreement 

                probs = F.softmax(positive_logits, dim=-1)
                most_likely = torch.argmax(positive_logits, dim=-1, keepdim=True)
                most_likely = torch.cat(most_likely, torch.ones(most_likely[0])) 
                all_agree = torch.all(most_likely == most_likely[0], dim=0) # 1 where all agree and is 1, 0 every where else


                pairwise_agreement = torch.zeros_like(positive_logits, dtype=torch.int)
                pairs = [(0,1), (0,2), (1,2)] ### If an agreeing pair is (0,1), would choose 2 (same for (0,2)). If (1,2), choose their output
                

                positive_logits = torch.sum(outs.logits[positive_adapters], dim=0) ### Base behavior for if there is no agreement. From here, add to this with the masks
                positive_logits += all_agree*1000 # Forcing all agree to be outputted

                if len(negative_adapters) == 0: ### Only doing pairwise if we have >2 adapters. 
                    for idx, (i,j) in enumerate(pairs):
                        agreement_ij = most_likely[i] == most_likely[j]

                        pairwise_agreement[idx] = (agreement_ij == 1) & (all_agree == 0) # Computing all pairwise agreements so that they are only 1 if they arent all agreeing 

                    positive_logits += pairwise_agreement[-1]*1000 # Forcing output if (1,2) agree


                    ### Need to create behavior for if (0,1) or (0,2) to choose the other adapter.

                if len(negative_adapters) > 0: 
                    pass
        
        return outs
    
    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model if self.active_peft_config.is_prompt_learning else self.base_model.model
    
    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def load_adapter(self, layer_idx, parameters, verbose=False):
        key_list = parameters.keys()
        for key in key_list:
            if '.lora_A' in key:
                layer_key = key.split("base_model.model.model.layers.")[-1]
                layer_num = layer_key.split(".")[0]

                slora_key = key.split('.lora_A')[0]
                _, target, _ = _get_submodules(self, slora_key)
                if isinstance(target, SloraLayer):
                    if verbose: print(f"Loading {key} to {slora_key}.lora_A[{layer_idx}][{self.active_adapter}]")
                    target.lora_A[layer_idx][self.active_adapter].weight = parameters[key]
                    target.lora_A[layer_idx][self.active_adapter].requires_grad_=False
                    target.layer_num = int(layer_num)
            if '.lora_B' in key:
                layer_key = key.split("base_model.model.model.layers.")[-1]
                layer_num = layer_key.split(".")[0]

                slora_key = key.split('.lora_B')[0]
                _, target, _ = _get_submodules(self, slora_key)
                if isinstance(target, SloraLayer):
                    if verbose: print(f"Loading {key} to {slora_key}.lora_B[{layer_idx}][{self.active_adapter}]")
                    target.lora_B[layer_idx][self.active_adapter].weight = parameters[key]
                    target.lora_B[layer_idx][self.active_adapter].requires_grad_=False
                    target.layer_num = int(layer_num)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]
    
class SloraExpander(torch.nn.Module):
    pass

class SloraMerger(torch.nn.Module):
    pass
                
                
