# Question-Level-Difficulty

## Data Folder Structure
```
Question-Level-Difficulty/
  main.py
  
  models/
    utils.py
    setting.py
    
    QLD/
      qld_only_text.py
      qld.py
      function.py
      
  data/
    data_loader.py
    DramaQA_v2.1/
      AnotherMissOh/
        parser.py
        train.tsv
        test.tsv
        val.tsv
        AnotherMissOhQA_train_set.json
        AnotherMissOhQA_val_set.json
        AnotherMissOhQA_test_set.json
        AnotherMissOh_script.json
        AnotherMissOh_images/
          $IMAGE_FOLDERS
```
## Model Flow

```
Time series images -> CNN backbone (ResNet50) -> Bi-LSTM -> pooling => u
Question + utterance - > Roberta -> CLS pooling => v 
u*, v* = AttnF(u, v)
(u, v, u*, v*) -> classifier -> loss
```

@ LSTM <br>
memory: 0.0, logic: 0.0 <br>
@ GRU <br>
memory: 0.0, logic: 0.0 <br>
@ Base Roberta <br>
memory: , logic: 77.2 <br>
