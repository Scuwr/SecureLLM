
echo For Table 1 results:

echo first two columns:
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123E  > T1.C1.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123  > T1.C2.log
echo column 3-7:
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  T1.C3.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  T1.C4.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  T1.C5.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  T1.C6.log
python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T1.C7.log

echo For Table 2 results:

echo first two columns:
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123E  > T2.C1.log
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123  > T2.C2.log
echo column 3-7:
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  T2.C3.log
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  T2.C4.log
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  T2.C5.log
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  T2.C6.log
python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T2.C7.log

echo For Table 3 results:

echo first two columns:
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123E  > T3.C1.log
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123  > T3.C2.log
echo column 3-7:
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLoraHub  >  T3.C3.log
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraSum  >  T3.C4.log
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraMax  >  T3.C5.log
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLogit  >  T3.C6.log
python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/6NF_obf/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T3.C7.log

