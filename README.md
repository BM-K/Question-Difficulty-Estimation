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
## Model
<img src='https://user-images.githubusercontent.com/55969260/114975390-27e20c80-9ebf-11eb-932c-b05ee0b2a1ce.png'>

## Model Flow

```
Time series images -> CNN backbone (ResNet50) -> Bi-LSTM -> pooling => u
Question + utterance - > Roberta -> CLS pooling => v 
u*, v* = AttnF(u, v)
(u, v, u*, v*) -> classifier -> loss
```
ACC, F1 score <br>
@ Base Roberta <br>
memory: 90.30, 78.67 | logic: 79.79, 78.90 <br>
@ Proposed Model (u, v)<br>
memory: 97.40, 94.44 | logic: 84.5, 84.18 <br>
@ Proposed Model (u, v, u*, v*) <br>
memory: 97.9, 95.1 | logic: 85.17, 85.22 <br>
