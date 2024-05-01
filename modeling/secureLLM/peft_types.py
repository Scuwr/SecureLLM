import enum

class PeftType_Slora(str, enum.Enum):
    SLORA_SUM = "slora_sum"
    SLORA_AVG = "slora_avg"
    SLORA_ABS_MAX_SUM = "slora_abs_max_sum"
    SLORA_MAX_SUM = "slora_max_sum"
    SLORA_MAX_ELEM = "slora_max_elem"
    SLORA_MAX_DIFF_ELEM = "slora_max_diff_elem"
    SLORA_CONCAT = "slora_concat"
    SLORA_LSE = "slora_lse"
    SLORA_SM = "slora_sm"
    SLORA_LOGIT = "slora_logit"
    SLORA_CONSENSUS = "slora_consensus"
    SLORA_LORAHUB = "slora_lorahub"
