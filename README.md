# Question Difficulty Estimation
- Video Turing Test
  - https://github.com/khu-nlplab/VTT-KHU-2021
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
  How did Haeyoung1 feel such surreal emotions?:
    Memory Level 3 | Logic Level 3 
  ```

## Results
<img src='https://user-images.githubusercontent.com/55969260/121448461-a7c5b800-c9d2-11eb-9d8e-b56e4ea8bcbc.png'>

## V3 Dataset
- logic
    - only text: 97.34
    - text + image: 97.88
- memory
    - only text: 99.64
    - text + image: 99.88
