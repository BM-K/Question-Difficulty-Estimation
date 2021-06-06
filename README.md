# Question Difficulty Estimation

- DramaQA - Memory and Logic complexity <br>
  - https://arxiv.org/abs/2005.03356 <br>
- TVQA - Custom question difficulty <br>
  - https://arxiv.org/abs/1809.01696

## Data Folder Structure
```
Question-Level-Difficulty/
  main.py
  
  models/
    utils.py
    setting.py
    
    QLD/
      qld_only_text.py
      qld_memory.py
      qld_logic.py
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

## Run
  ```
  python main.py
  
  How did Haeyoung1 feel such surreal emotions?:
    Memory Level 3 | Logic Level 3 
  ```

## Results
<img src='https://user-images.githubusercontent.com/55969260/120881189-7469de80-c60a-11eb-91fb-0f1b92ce317c.png'>
