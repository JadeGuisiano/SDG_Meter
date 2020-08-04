# SDG_Meter 

A myriad of textual documents produced/consumed by UNEP need to be mapped to SDGs (project proposals, reports, briefings, etc.). Such mapping exercises demand extensive expert time and rely on personal knowledge of interlinkages among topics and SDGs. While UNEP counts with experts in several topics, interlinkages with SDGs outside our expertise can be missed out. 

That's why we propose an automatic text classification which for a given text permits to establish relationships with the SDGs treats in the text but also quantifies the degree to which the text belongs to each of the detected SDGs. This 


## Our repository contains: 

- (1) Our entire solution code is available in the Jupyter Notebook file "SDG_Meter_BERT_Algorithm.ipynb" but also is Python script file .py
- (2) Initial/short database "SDG_Objectives.csv" in CSV format
- (3) All the process/algorithm/API we used to clean and improve the orignal database "Database_conception_BERT.ipynb"
- (4) The final database obtained thanks to steps in (3) "res.csv" in CSV format
- (5) All the required packages for your environment (obtained via pip freeze) "requirements.rtf"


## /!\ important to know before launching /!\

- Concerning part (1), you need to download BERT pre-trained model and place it in the same repository that all mentioned files (1) to (5) in this link: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

- In order to install all the requirement packages, you need to first go on your repository on shell and then run "pip install -r requirements.rtf"


## Algorithm detailed 

(1) Our method consists in a multilabel classification (because we want to be able to link a text not just to one SDG but all contained in the text)
To realize this task, we use BERT [1] model which permits to obtain the probabilities that SDGs belong to a text.

(3) Our initial database contains targets for each SDG, in order to have another style than UN institutional we decide to add synonymes of each SDG keyword and get its definition by Wikipedia API, we also integrate PDF text information than was provided to us by UN expert for that we need to extract text from PDF. Finally we use Makvofivy (Markov chain for text generation) in order to obtain 300 extra rows for each SDG.






[1] Devlin, Jacob et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” NAACL-HLT (2019).
