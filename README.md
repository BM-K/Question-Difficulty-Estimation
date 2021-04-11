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
ACC, F1 score <br>
@ Base Roberta <br>
memory: 90.9 , 81.36 | logic: 77.2, 77.2 <br>
90.30, 78.67 | 79.79, 78.90 <br><br>
@ Proposed Model (u, v)<br>
memory: 97.6, 94.3 | logic: 84.3, 84.3 <br>
97.40, 94.44 |
@ Proposed Model (u, v, u*, v*) <br>
memory: 97.7, 94.50 | logic: 85.5, 85.5 <br><br>
97.9, 95.1 | 85.17, 85.22
