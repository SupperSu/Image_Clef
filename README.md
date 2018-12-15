# Model for Image CLEF competition.
Model is based on Residual-Net101 and LSTM medical image caption, get bleu score 0.20, trained on 222,000 images on amazon ec2.  
TODO: refactor codes, and redesign attention module to use medical concepts.
## sample output
![](/brain.jpg)  
**prediction:** ct scan showing a large retroperitoneal hematoma with associated left hydronephrosis  
**ground truth:** retropancreatic tumor a preoperative ct scan showing the large retropancreatic tumor  
![](/blood.jpg)  
**prediction:** photomicrograph of the resected specimen showing a well encapsulated tumor with a thin wall of fibrous tissue hematoxylin and eosin.  
**ground truth:** liver section of female test rat treated with aqueous extract of p guajava leaves cords of hepatocytes are distinct and relatively normal no fatty change cytoplasm not vacuolated haematoxylin and eosin stained.  
