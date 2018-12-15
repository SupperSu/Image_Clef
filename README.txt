The dataset files for the two Caption tasks are structured as follows:

======================
Task 1 - CaptionPrediction:
======================
The 223859 training images are in CaptionTraining2018, which was uploaded as split zip files:
	"CaptionTraining2018.zip.aa"
	"CaptionTraining2018.zip.ab"
	"CaptionTraining2018.zip.ac"
	"CaptionTraining2018.zip.ad"	
	"CaptionTraining2018.zip.ae"
	"CaptionTraining2018.zip.af"
Use 'cat CaptionTraining2018.zip.* > CaptionTraining2018.zip' 
then 'unzip CaptionTraining2018.zip' to extract this folder.

"CaptionPredictionTraining2018-Captions.csv" contains the training images names with their associated captions.
"CaptionTraining2018-List.txt" contains the list of training images names.

======================
Task 2 - ConceptDetection:
======================
The training images are the same as Task 1 in CaptionTraining2018.

"ConceptDetectionTraining2018-Concepts.csv" contains the training images names with their associated concepts.
"CaptionTraining2018-List.txt" contains the list of training images names (same as Task 1).


111156 concepts.

