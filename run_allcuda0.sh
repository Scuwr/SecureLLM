mkdir logs

echo "exposing only cuda 0 to huggingface"
export CUDA_VISIBLE_DEVICES=0

echo For Table 1 results:

echo first two columns:
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123E  > logs/T1.C1.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123  > logs/T1.C2.log
echo column 3-7:
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  logs/T1.C3.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  logs/T1.C4.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  logs/T1.C5.log
# python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  logs/T1.C6.log
# python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  logs/T1.C7.log

# echo For Table 2 results:

# echo first two columns:
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123E  > logs/T2.C1.log
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123  > logs/T2.C2.log
# echo column 3-7:
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  logs/T2.C3.log
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  logs/T2.C4.log
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  logs/T2.C5.log
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  logs/T2.C6.log
# python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  logs/T2.C7.log

# echo For Table 3 results:

# echo first two columns:
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123E  > logs/T3.C1.log
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123  > logs/T3.C2.log
# echo column 3-7:
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLoraHub  >  logs/T3.C3.log
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraSum  >  logs/T3.C4.log
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraMax  >  logs/T3.C5.log
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLogit  >  logs/T3.C6.log
# python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/6NF_obf/" --pseudo --models M1 M2 M3 --config SloraLogit  >  logs/T3.C7.log

